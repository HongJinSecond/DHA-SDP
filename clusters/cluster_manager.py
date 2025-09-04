import torch
from utils.process_datasets import ClusterDataset, JITFineDataset
from sklearn.ensemble import IsolationForest


class ClusterManager(object):
    def __init__(self, args, cluster_model, lora_dir:str):
        self.args = args
        self.cluster_model = cluster_model
        self.lora_dir = lora_dir
        # 方法1: 孤立森林 (推荐)
        self.isolation_forest_outliers=dict()
        self.trainFeatures=dict()
        self.trainDatasets=dict()

    def set_cluster_model(self, cluster_model):
        self.cluster_model = cluster_model


    def _outlierPredictorInit(self):
        for key,train in self.trainFeatures.items():
                clf = IsolationForest(
                    contamination=0.1,
                    random_state=self.args.seed
                )
                clf.fit(train)
                self.isolation_forest_outliers[key]=clf

    def init_cluster_model(self,train_examples,features):
        # features=self.cluster_model.extract_features(train_examples)
        self.cluster_model.fit_clusters(train_examples,features)
        self.trainDatasets=self.split_datasets(train_examples)
        self._outlierPredictorInit()

    def _init_lora_dir(self,path):
        self.lora_dir = path

    def predict_clusters(self,features):
        """

        :param features: Only manual features for predicting cluster label (N , 14):[[......],[......],......]
        :return: classes (-1,0,......,n_classes)
        """
        print("Predicting clusters...")
        labels=self.cluster_model.predict_class_only_for_features(features)
        print(f"Done! labels: {labels}")
        return labels


    def split_datasets(self,examples):
        """
        Split dataset into multiple lora datasets
        :param original examples:
        :return: {ClusterDataset}
        """
        datasets_dict={}
        features=self.extract_manual_features(examples)
        # features = self.cluster_model.extract_features(examples)
        clusters = self.cluster_model.predict_cluster(examples,features)
        print("\nCluster distribution with anomalies:")
        for label, items in clusters.items():
            datasets = ClusterDataset(items,label)
            datasets_dict[label]=datasets
            self.trainFeatures[label]=self.extract_manual_features(datasets)
        return datasets_dict

    def extract_manual_features(self,examples):
        return self.cluster_model.extract_features(examples)
    
    def __isolation(self,test_sample,lora_key):
        clf=self.isolation_forest_outliers[lora_key]
        return clf.predict(test_sample) == -1  # -1表示异常
    
    def checkOutliers(self,test_sample,lora_key,strategy="Isolation")->bool:
        assert strategy in ["Isolation","Lof","Mahalanobis","tri"],"Invalid commands!"
        if strategy=="Isolation":
            return self.__isolation(test_sample,lora_key)

if __name__ == "__main__":
    ######################### just for test ###############################
    from feature_cluster import FeatureClusterProcessor

    cluster_model = FeatureClusterProcessor(n_clusters=8)

    from utils.process_datasets import ClusterDataset,JITFineDataset
    from utils.util import *
    args = parse_jit_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)
    train_dataset = JITFineDataset(tokenizer,args,"train")
    eval_dataset = JITFineDataset(tokenizer, args, "eval")
    print(f"eval dataset have {len(eval_dataset)} samples")
    test_dataset = JITFineDataset(tokenizer, args, "test")
    print(f"test dataset have {len(test_dataset)} samples")

    examples = train_dataset.examples

    cluster_manager=ClusterManager(args, cluster_model, "LORA")
    cluster_manager.init_cluster_model(examples)

    split_datasets_train=cluster_manager.split_datasets(examples)
    ############################# Split Datasets ####################################
    print("train datasets:")
    for label,clusterDataset in split_datasets_train.items():
        print(f"{label}: {len(clusterDataset)} samples")

    # print("use dataset for lora 2")
    # from torch.utils.data import DataLoader
    #
    # loader=DataLoader(eval_dataset, batch_size=10, shuffle=True)
    # for Input_ids,mask,features,y in loader:
    #     print(Input_ids.shape,mask.shape,features.shape,y.shape)
    #     break

    ############################ Split Eval Datasets #######################################
    split_datasets_eval = cluster_manager.split_datasets(eval_dataset)
    print("split eval dataset")
    for label,clusterDataset in split_datasets_eval.items():
        print(f"{label}: {len(clusterDataset)} samples")

    ############################ Split test Datasets #######################################
    split_datasets_test = cluster_manager.split_datasets(test_dataset)
    print("split test dataset")
    for label,clusterDataset in split_datasets_test.items():
        print(f"{label}: {len(clusterDataset)} samples")

    # ############################ train test ##############################
    # print("find the best data for it")
    # for label,clusterDataset in split_datasets_train.items():
    #     eval_lora_datasets = split_datasets_eval.get(label)
    #     if eval_lora_datasets is None:
    #         print(f"eval {label} : None")
    #     else:
    #         print(f"eval {label} : {len(eval_lora_datasets)} samples")
    #
    # special_data = split_datasets_eval.get(-1)
    # print(f"eval -1 : {len(special_data)} samples")