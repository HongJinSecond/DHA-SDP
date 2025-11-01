from clusters.cluster_manager import ClusterManager
from clusters.HierarchicalCluster import HierarchicalCluster
from clusters.feature_cluster import FeatureClusterProcessor

from utils.util import *
from utils.process_datasets import load_developer_datas,load_project_datas,JITFineDataset

import json


CLUSTER_MODEL_LIST = ["developer_aware","Kmean","project_final"]
N_CLUSTER = 4
BASE_MODEL = "concat"


if __name__ == "__main__":
    all_dict={}
    args=parse_jit_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])
    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    for cluster in CLUSTER_MODEL_LIST:
        print(f"run {cluster}")
        if cluster=="developer_aware":
            developer_dict={}
            ####################################### Load datasets ######################################################
            train_dataset_dict = load_developer_datas(tokenizer, args, "train")
            test_dataset_dict = load_developer_datas(tokenizer, args, "test")
            valid_dataset_dict = load_developer_datas(tokenizer,args,"eval")
            fcluster=HierarchicalCluster(args,n_cluster=args.n_cluster)
            fcluster.fit(train_dataset_dict)
            ####################################### split datasets #####################################################
            train_datasets=fcluster.splitDatasets(train_dataset_dict)
            test_datasets=fcluster.splitDatasets(test_dataset_dict)
            valid_datasets=fcluster.splitDatasets(valid_dataset_dict)
            
            ###################################### counter ##########################################
            train_dict={}
            for key,values in train_datasets.items():
                train_dict[str(key)]=len(values)

            test_dict={}
            for key,values in test_datasets.items():
                test_dict[str(key)]=len(values)

            valid_dict={}
            for key,values in valid_datasets.items():
                valid_dict[str(key)]=len(values)

            developer_dict["train"]=train_dict
            developer_dict["valid"]=valid_dict
            developer_dict["test"]=test_dict
            all_dict["developer_aware"]=developer_dict

        elif cluster=="project_final":
            project_dict={}
            ####################################### Load datasets ######################################################
            train_dataset_dict = load_project_datas(tokenizer, args, "train")
            test_dataset_dict = load_project_datas(tokenizer, args, "test")
            valid_dataset_dict = load_project_datas(tokenizer,args,"eval")
            fcluster=HierarchicalCluster(args,n_cluster=args.n_cluster)
            fcluster.fit(train_dataset_dict)
            ####################################### split datasets #####################################################
            train_datasets=fcluster.splitDatasets(train_dataset_dict)
            test_datasets=fcluster.splitDatasets(test_dataset_dict)
            valid_datasets=fcluster.splitDatasets(valid_dataset_dict)
            
            ###################################### counter ##########################################
            train_dict={}
            for key,values in train_datasets.items():
                train_dict[str(key)]=len(values)

            test_dict={}
            for key,values in test_datasets.items():
                test_dict[str(key)]=len(values)

            valid_dict={}
            for key,values in valid_datasets.items():
                valid_dict[str(key)]=len(values)

            project_dict["train"]=train_dict
            project_dict["valid"]=valid_dict
            project_dict["test"]=test_dict
            all_dict["project_aware"]=project_dict

        elif cluster=="Kmean":
            Kmean_dict={}

            train_dataset = JITFineDataset(tokenizer, args, "train")
            test_dataset = JITFineDataset(tokenizer, args, "test")
            valid_dataset = JITFineDataset(tokenizer, args, "eval")
            cluster_model = FeatureClusterProcessor(args,n_clusters=args.n_cluster)
            cluster_manager = ClusterManager(args, cluster_model,"")

            train_features = cluster_model.extract_features(train_dataset)
            ############################################################################################################
            cluster_manager.init_cluster_model(train_dataset,train_features)


            test_datasets=cluster_manager.split_datasets(test_dataset)
            valid_datasets=cluster_manager.split_datasets(valid_dataset)
            train_datasets = cluster_manager.cluster_model.clustered_data

            train_dict={}
            for key,values in train_datasets.items():
                train_dict[str(key)]=len(values)

            test_dict={}
            for key,values in test_datasets.items():
                test_dict[str(key)]=len(values)

            valid_dict={}
            for key,values in valid_datasets.items():
                valid_dict[str(key)]=len(values)

            Kmean_dict["train"]=train_dict
            Kmean_dict["valid"]=valid_dict
            Kmean_dict["test"]=test_dict
            all_dict["Kmean"]=Kmean_dict

        else:
            print("错误！！！！！")
    



    # 方法1：保存到JSON文件
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(all_dict, f, ensure_ascii=False, indent=4)

    # # 方法2：转换为JSON字符串
    # json_string = json.dumps(data, ensure_ascii=False, indent=4)
    # print(json_string)
                    
