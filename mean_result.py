import pandas as pd
import numpy as np
import os

MODEL_NAME_LIST = ["codebert", "graphcodebert", "unixcoder", "plbart", "codet5"]
CLUSTER_MODEL_LIST = ["project_fianl", "developer_aware", "Kmean"]
N_CLUSTER = 4
BASE_MODEL = "concat"

for cluster_model in CLUSTER_MODEL_LIST:
    for model_name in MODEL_NAME_LIST:
        # 初始化列表存储五次实验结果
        base_results_list = []
        lora_results_list = []
        outlier_results_list = []
        
        ####################### 获取五次实验的结果 ##########################
        for result_time in ["result0", "result1", "result2", "result3", "result4"]:
            base_path = os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/{model_name}/{result_time}", "base_results.csv")
            lora_path = os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/{model_name}/{result_time}", "lora_results.csv")
            outlier_path = os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/{model_name}/{result_time}", "outlier_results.csv")

            base_result = pd.read_csv(base_path, index_col=0)  # 第一列作为索引
            lora_result = pd.read_csv(lora_path, index_col=0)
            outlier_result = pd.read_csv(outlier_path, index_col=0)
            
            # 将结果添加到列表
            base_results_list.append(base_result)
            lora_results_list.append(lora_result)
            outlier_results_list.append(outlier_result)

        # 计算平均值
        base_mean = pd.concat(base_results_list).groupby(level=0).mean()
        lora_mean = pd.concat(lora_results_list).groupby(level=0).mean()
        outlier_mean = pd.concat(outlier_results_list).groupby(level=0).mean()

        # 创建保存平均结果的目录
        avg_dir = os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/{model_name}", "average")
        os.makedirs(avg_dir, exist_ok=True)

        # 保存平均结果
        base_mean.to_csv(os.path.join(avg_dir, "base_results.csv"))
        lora_mean.to_csv(os.path.join(avg_dir, "lora_results.csv"))
        outlier_mean.to_csv(os.path.join(avg_dir, "outlier_results.csv"))