import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from utils.process_datasets import JITFineDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class FeatureClusterProcessor:
    def __init__(self,args, cluster_model=None, n_clusters=5, threshold_factor=1.2,method="pca"):
        """
        初始化聚类处理器
        :param cluster_model: 聚类模型实例（需实现fit_predict和predict方法），默认使用KMeans
        :param n_clusters: 当使用默认模型时的聚类数量
        :param threshold_factor: 异常检测阈值系数（基于聚类最大距离的倍数）
        """
        self.cluster_model = cluster_model or KMeans(n_clusters=n_clusters,init='k-means++',random_state=args.seed)
        self.n_clusters = n_clusters
        self.cluster_labels = None
        self.clustered_data = None
        self.args = args
        self.cluster_centers_ = None
        self.method = method.lower()
        self.thresholds_ = None  # 每个聚类的距离阈值
        self.threshold_factor = threshold_factor
        self.reducer = None
        self.fitted = False
        self.n_clusters = n_clusters

    def save(self, filepath):
        """
        将整个处理器对象保存到文件
        :param filepath: 保存路径（推荐使用.pkl后缀）
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.cluster_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        """
        从文件加载处理器对象
        :param filepath: 保存路径
        :return: 加载后的FeatureClusterProcessor实例
        """
        with open(filepath, 'rb') as f:
            self.cluster_model = pickle.load(f)


    def extract_features(self, data,cluster_target=3):
        """从原始数据中提取特征矩阵"""
        return torch.stack([item[cluster_target] for item in data], dim=0).numpy()

    def _calculate_thresholds(self, features):
        """计算每个聚类的距离阈值"""
        distances = pairwise_distances(features, self.cluster_centers_)
        cluster_distances = defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            cluster_distances[label].append(distances[idx, label])

        self.thresholds_ = {}
        for label, dists in cluster_distances.items():
            max_dist = np.max(dists)
            self.thresholds_[label] = max_dist * self.threshold_factor

    def fit_clusters(self, data , features):
        """
        训练聚类模型并划分数据
        :param data: 原始数据集[A1, A2, A3,...]
        """
        # features = self._extract_features(data)
        self.cluster_labels = self.cluster_model.fit_predict(features)
        self.cluster_centers_ = self.cluster_model.cluster_centers_

        # 计算距离阈值
        self._calculate_thresholds(features)

        # 按聚类结果分组数据
        self.clustered_data = defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            self.clustered_data[label].append(data[idx])
        self.clustered_data = dict(self.clustered_data)

        # 降维 for cluster display
        if self.method == 'pca':
            self.reducer = PCA(n_components=2)
        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=2)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        self.embedding = self.reducer.fit_transform(features)
        self.fitted = True

        return self

    def predict_class(self, new_features):
        """
        新接口：直接返回每个样本的聚类类别
        :return: 包含类别标签的列表（异常类为-1）
        """
        # 计算到所有聚类中心的距离
        distances = pairwise_distances(new_features, self.cluster_centers_)

        pred_labels = []
        for i in range(len(new_features)):
            # 找到最近聚类及其距离
            nearest_label = np.argmin(distances[i])
            pred_labels.append(nearest_label)

        return pred_labels

    def predict_class_only_for_features(self,features):
        # 计算到所有聚类中心的距离
        features=features.numpy()
        distances = pairwise_distances(features, self.cluster_centers_)

        pred_labels = []
        for i in range(len(features)):
            # 找到最近聚类及其距离
            nearest_label = np.argmin(distances[i])
            min_distance = distances[i, nearest_label]

            # 检查是否超过阈值
            if min_distance > self.thresholds_.get(nearest_label, np.inf):
                pred_labels.append(-1)
            else:
                pred_labels.append(int(nearest_label))

        return pred_labels

    def predict_cluster(self, new_data,new_feature):
        """
        改进的预测方法：包含异常类(-1)
        :return: 按聚类分组的新数据字典（包含-1异常类）
        """
        pred_labels = self.predict_class(new_feature)

        new_clusters = defaultdict(list)
        for idx, label in enumerate(pred_labels):
            new_clusters[label].append(new_data[idx])

        return dict(new_clusters)

    # 以下原有方法保持不变
    def get_cluster(self, label):
        return self.clustered_data.get(label, [])

    def get_all_clusters(self):
        return self.clustered_data
    
    def clean_datasets(self,datasets,features):
        """
        过滤原始数据集的噪声
        :return:
        """
        # 处理离群异常点
        filter = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.args.seed)
        filter.fit(features)
        labels = filter.labels_

        # 检查点到中心的距离
        distances = np.linalg.norm(features - filter.cluster_centers_[labels], axis=1)
        outlier_mask = distances > np.percentile(distances, 80)  # 移除前20%的远距离点
        final_clean_feature = features[~outlier_mask]
        datasets.examples = [item for item, flag in zip(datasets.examples, ~outlier_mask) if flag]
        print(len(datasets.examples), len(final_clean_feature))
        return datasets,final_clean_feature



    def visualize(self, X_test=None, title="Clustering Visualization"):
        """
        可视化聚类结果，可同时显示训练集和测试集
        
        参数:
        - X_test: 测试数据 (n_samples, n_features)，可选
        - title: 图表标题
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
            
        plt.figure(figsize=(10, 8))
        
        # 为每个聚类创建颜色
        colors = list(mcolors.TABLEAU_COLORS.values())
        if self.n_clusters > len(colors):
            colors = list(mcolors.CSS4_COLORS.values())
        
        # 绘制训练集(总是显示)
        for i in range(self.n_clusters):
            mask = (self.cluster_model.labels_ == i)
            plt.scatter(
                self.embedding[mask, 0], 
                self.embedding[mask, 1],
                color=colors[i],
                alpha=0.3
                ,  # 训练集用浅色
                label=f'Train Cluster {i}' if X_test is None else f'Cluster {i}',
                s=40  # 点大小
            )
        
        # 如果有测试集，绘制测试集
        test_labels = None
        if X_test is not None:
            # 标准化并降维
            test_embedding = self.reducer.transform(X_test)
            
            # 预测聚类标签
            test_labels = np.array(self.predict_class(X_test))
            
            # 绘制测试集的离群点
            mask = (test_labels == -1)
            plt.scatter(
                test_embedding[mask, 0], 
                test_embedding[mask, 1],
                color="black",
                edgecolors='black',
                linewidths=0.8,
                label=f'Test Outliers',
                alpha=0.8,
                s=60  # 测试集点稍大
            )
            # 绘制测试集点
            for i in range(self.n_clusters):
                mask = (test_labels == i)
                plt.scatter(
                    test_embedding[mask, 0], 
                    test_embedding[mask, 1],
                    color=colors[i],
                    edgecolors='black',
                    linewidths=0.8,
                    label=f'Test Cluster {i}',
                    alpha=0.8,
                    s=60  # 测试集点稍大
                )
                
        # 绘制聚类中心
        centers_2d = self.reducer.transform(self.cluster_model.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            marker='x', s=200, linewidths=3,
            color='black', label='Cluster Centers'
        )
        
        plt.title(title)
        plt.xlabel(f"{self.method.upper()} Component 1")
        plt.ylabel(f"{self.method.upper()} Component 2")
        
        # 优化图例显示
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return test_labels





