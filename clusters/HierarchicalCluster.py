import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster
from scipy.stats import wasserstein_distance
from torch.utils.data import ConcatDataset

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.svm import OneClassSVM

from scipy.stats import chi2

def pairwise_wasserstein(collections):
    """Wasserstein distance between datasets"""
    n = len(collections)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Use the mean wasserstein distance among the 14 features.
            dists = [wasserstein_distance(collections[i][:,d], collections[j][:,d]) 
                     for d in range(collections[i].shape[1])]
            dist_matrix[i,j] = np.mean(dists)
            dist_matrix[j,i] = dist_matrix[i,j]
    return squareform(dist_matrix)

def extract_expert_features(data,cluster_target=3):
    """extract expert features from JITFine Dataset"""
    return np.stack([item[cluster_target] for item in data], axis=0)


class HierarchicalCluster:
    def __init__(self,args,n_cluster=4):
        self.n_cluster=n_cluster
        self.trainFeatures=dict()
        self.args=args
        self.trainKeys=None
        self.mapping_dict=None

        # 方法1: 孤立森林 (推荐)
        self.isolation_forest_outliers=dict()

        # 方法2: 局部离群因子(LOF)
        self.lof_outliers=dict()

        # 方法3: 马氏距离
        self.mahalanobis_outliers=dict()

        # 方法4：one class SVM
        self.svms=dict()


    def _trainFeatureSplit(self,collections,cluster):
        for i in range(self.n_cluster):
            self.trainFeatures[i+1]=[]
        for k,features in zip(cluster,collections):
            self.trainFeatures[int(k)].append(features)
        for i in range(self.n_cluster):
            self.trainFeatures[i+1]=np.concatenate(self.trainFeatures[i+1],axis=0)


    def _outlierPredictorInit(self):
        for key,train in self.trainFeatures.items():
                clf = IsolationForest(
                    contamination=0.1,
                    random_state=self.args.seed
                )
                clf.fit(train)

                lof = LocalOutlierFactor(
                    contamination="auto", 
                    novelty=True
                )
                lof.fit(train)

                svm=OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
                svm.fit(train)


                robust_cov = MinCovDet(random_state=self.args.seed,support_fraction=0.8).fit(train)
                
                self.isolation_forest_outliers[key]=clf
                self.lof_outliers[key]=lof
                self.svms[key]=svm
                self.mahalanobis_outliers[key]=robust_cov

    def fit(self,trainDatasets):
        self.trainKeys=list(trainDatasets.keys())

        # Collecting the features for each dataset
        collections=[]

        for name,dataset in trainDatasets.items():
            collections.append(extract_expert_features(dataset))
        # Hierachical Cluster
        distances=pairwise_wasserstein(collections)
        Z=linkage(distances,method="average")
        self.Z=Z
        self.clusters=fcluster(Z,t=self.n_cluster,criterion="maxclust")
        ### Mapping orignial to new cluster
        self._trainFeatureSplit(collections,self.clusters)
        self._outlierPredictorInit()
        cluster_dict=dict()
        for i,key_name in enumerate(self.trainKeys):
            cluster_dict[key_name]=int(self.clusters[i])
        self.mapping_dict=cluster_dict

    def splitDatasets(self,targetDatasets):
        """
        The original dataset would be splited into multiple datasets for Loras (Both for train, test and valid).
        :param targetDatasets:
        :return:
        """
        # split the datasets by clustering
        cluster_dict=dict()
        outliers=[]
        datasets_list=[[] for _ in range(self.n_cluster)]
        for key,item in targetDatasets.items():
            if key in self.mapping_dict.keys():
                # the train lora contains this project 
                datasets_list[self.mapping_dict[key]-1].append(item)
            else:
                # train lora does not contains this project
                outliers.append(item)
        for i in range(len(datasets_list)):
            # Not empty
            if datasets_list[i]!=[]:
                cluster_dict[i+1]=ConcatDataset(datasets_list[i])
        # add outlier datasets . only for test case
        if outliers!=[]:
            cluster_dict["other"]=ConcatDataset(outliers)
        return cluster_dict

    def checkOutliers(self,test_sample,lora_key,strategy="Mahalanobis")->bool:
        """
        This function will check if one sample is outlier or not.
        :param targetDatasets:
        :return:
        """
        assert strategy in ["Isolation","Lof","Mahalanobis","tri"],"Invalid commands!"
        if strategy=="Isolation":
            return self.__isolation(test_sample,lora_key)

        elif strategy=="Lof":
            return self.__Lof(test_sample,lora_key)
        
        elif strategy=="svm":
            return self.__one_class_SVM(test_sample,lora_key)
        
        elif strategy=="Mahalanobis":
            return self.__mahalanobis(test_sample,lora_key)
        
        elif strategy=="tri":
            # 满足至少两种指标才算离群点
            isolation=self.__isolation(test_sample,lora_key)
            svm=self.__one_class_SVM(test_sample,lora_key)
            mahalanobis=self.__mahalanobis(test_sample,lora_key)
            return np.sum(np.stack([isolation,svm,mahalanobis],axis=-1),axis=-1)>=2
        
        else:
            print("Invalid command, no outlier adjustmemt!")
            return None

    def __isolation(self,test_sample,lora_key):
        clf=self.isolation_forest_outliers[lora_key]
        return clf.predict(test_sample) == -1  # -1表示异常
    
    def __Lof(self,test_sample,lora_key):
        lof=self.lof_outliers[lora_key]
        return lof.predict(test_sample) == -1
    
    def __one_class_SVM(self,test_sample,lora_key):
        svm=self.svms[lora_key]
        return svm.predict(test_sample)==-1
    
    def __mahalanobis(self,test_sample,lora_key):
        robust_cov=self.mahalanobis_outliers[lora_key]
        mahal_dist = robust_cov.mahalanobis(test_sample)
        
        # 计算阈值 - 使用卡方分布
        n_features = 14
        threshold = chi2.ppf(0.9999, df=n_features)
    
        # 标记离群点
        outliers = mahal_dist > threshold
        return outliers
