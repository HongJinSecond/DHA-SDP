import torch
import logging
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve,matthews_corrcoef
from utils.util import effort_aware_metrics
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from models.FinalModel import FinalModelWeight as FinalModel
from utils.util import parse_jit_args, set_seed, build_model_tokenizer_config,get_peft_lora_config,ensure_directory_exists,write_test_results_to_csv
from utils.process_datasets import load_project_datas
from models.SingleModel import SingleModel
from models.ConcatModel import ConcatModel
from models.ManualModel import ManualModel
from config import *
from project_style_aware import main as train_lora
from clusters.HierarchicalCluster import HierarchicalCluster

from weightTrainer import weight_calculator

logger = logging.getLogger(__name__)
import pandas as pd
import matplotlib.pyplot as plt

columns = ["project", "accuracy", "precision", "recall", "f1", "gmean", "mcc", "auc_score"]



def calculate_metrics(pred_prob,pred_label,true_label):

    fpr, tpr, thres = roc_curve(true_label, pred_prob)
    auc_score = auc(fpr, tpr)
    metrics = {
        "accuracy": (pred_label == true_label).mean(),
        "precision": precision_score(true_label, pred_label, average="binary"),
        "recall": recall_score(true_label, pred_label, average="binary"),
        "recall0": recall_score(true_label, pred_label, pos_label=0, average="binary"),
        "f1": f1_score(true_label, pred_label, average="binary"),
        "gmean": np.sqrt(
            recall_score(true_label, pred_label, pos_label=0, average="binary") *
            recall_score(true_label, pred_label, average="binary")
        ),
        "mcc": matthews_corrcoef(true_label, pred_label),
        "auc":auc_score
        
    }

    return metrics


def compare_test(args, test_dataset_dict, base_model,cluster_manager:HierarchicalCluster,weight_dict:dict):
    #load model
    base_model.eval()
    finalModel = FinalModel(args, base_model)
    finalModel.eval()

    #prediction result for base case and lora-used case
    global_pred_prob_base = []
    global_pred_prob_lora = []
    global_pred_prob_outlier=[]
    global_true_label = []

    #results table
    base_results=[]
    lora_results=[]
    outlier_results=[]


    ensure_directory_exists(os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}"))


    for project_name, test_cluster_datasets in test_dataset_dict.items():
        # local prediction case
        local_pred_prob_base = []
        local_pred_prob_lora = []
        local_pred_prob_outlier=[]
        local_true_label = []

        commit_hashes=[]

        # print(type(project_name))
        print(f"Project: {project_name}: {len(test_cluster_datasets)} samples.")
        test_sampler = SequentialSampler(test_cluster_datasets)
        test_dataloader = DataLoader(test_cluster_datasets, sampler=test_sampler, batch_size=args.batch_size)
        # load lora
        finalModel.load_lora(
            os.path.join(LORA_DIR, f"{args.cluster_model}/{args.n_cluster}/{args.base_model}/{args.pretrained_model}/{project_name}"),
            lora_name=str(project_name))

        for batch in test_dataloader:
            # check outlier samples
            outlier_mask=cluster_manager.checkOutliers(batch[3].numpy(),lora_key=project_name,strategy=args.strategy)
            outlier_mask=torch.tensor(outlier_mask).to(args.device)
            if outlier_mask.sum()==0:
                print("No outlier.")
                outlier_mask=None
            commit_hash, input_ids, input_mask, manual_features, label = batch
            input_ids, input_mask, manual_features, label = (
                x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
            )
            commit_hashes.extend(list(commit_hash))
            with torch.no_grad():
                ###############store the true case
                global_true_label.append(label.detach().cpu().numpy())
                local_true_label.append(label.detach().cpu().numpy())
                ###############RUN for lora with mask
                finalModel.change_loras(str(project_name))
                weight=weight_dict.get(project_name)
                # print(f"get weight: {weight}")
                prob, loss = finalModel(input_ids, input_mask, manual_features, label,use_base=False,outlier_mask=outlier_mask,weight=weight)
                global_pred_prob_outlier.append(prob.detach().cpu().numpy())
                local_pred_prob_outlier.append(prob.detach().cpu().numpy())
                ###############Run for lora without mask
                prob, loss = finalModel(input_ids, input_mask, manual_features, label,use_base=False)
                global_pred_prob_lora.append(prob.detach().cpu().numpy())
                local_pred_prob_lora.append(prob.detach().cpu().numpy())
                ###############run for base model
                prob, loss = finalModel(input_ids, input_mask, manual_features, label,use_base=True)
                global_pred_prob_base.append(prob.detach().cpu().numpy())
                local_pred_prob_base.append(prob.detach().cpu().numpy())

        #######################计算并保存lora模型在特定数据集上的表现性能，待完善
        # 计算当前数据集指标
        local_pred_prob_lora = np.concatenate(local_pred_prob_lora, 0)
        local_pred_prob_base = np.concatenate(local_pred_prob_base, 0)
        local_pred_prob_outlier=np.concatenate(local_pred_prob_outlier,0)

        local_true_label = np.concatenate(local_true_label, 0)
        # 指标计算
        best_threshold = args.threshold
        pred_label_lora = [0 if x < best_threshold else 1 for x in local_pred_prob_lora]
        pred_label_base = [0 if x < best_threshold else 1 for x in local_pred_prob_base]
        pred_label_outlier = [0 if x < best_threshold else 1 for x in local_pred_prob_outlier]

        # print(pred_label_lora==pred_label_base)
        # 记录结果
        lora_results.append({
            "project": project_name,
            **calculate_metrics(local_pred_prob_lora,pred_label_lora,local_true_label)
        })
        base_results.append({
            "project": project_name,
            **calculate_metrics(local_pred_prob_base,pred_label_base,local_true_label)
        })
        outlier_results.append({
            "project": project_name,
            **calculate_metrics(local_pred_prob_outlier,pred_label_outlier,local_true_label)
        })


        write_test_results_to_csv(commit_ids=commit_hashes,
                            probs=local_pred_prob_base.reshape(-1),
                            prediction_labels=pred_label_base,
                            true_labels=local_true_label,
                            cluster=project_name,
                            filename=os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/base_output.csv"))

        write_test_results_to_csv(commit_ids=commit_hashes,
                            probs=local_pred_prob_lora.reshape(-1),
                            prediction_labels=pred_label_lora,
                            true_labels=local_true_label,
                            cluster=project_name,
                            filename=os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/lora_output.csv"))

        write_test_results_to_csv(commit_ids=commit_hashes,
                            probs=local_pred_prob_outlier.reshape(-1),
                            prediction_labels=pred_label_outlier,
                            true_labels=local_true_label,
                            cluster=project_name,
                            filename=os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/outlier_output.csv"))
        
    global_pred_prob_lora = np.concatenate(global_pred_prob_lora, 0)
    global_pred_prob_base = np.concatenate(global_pred_prob_base, 0)
    global_pred_prob_outlier = np.concatenate(global_pred_prob_outlier, 0)


    global_true_label = np.concatenate(global_true_label, 0)
    best_threshold = args.threshold
    pred_label_lora = [0 if x < best_threshold else 1 for x in global_pred_prob_lora]
    pred_label_base = [0 if x < best_threshold else 1 for x in global_pred_prob_base]
    pred_label_outlier= [0 if x < best_threshold else 1 for x in global_pred_prob_outlier]

    test_expert_features=pd.read_pickle(args.test_data_file[1])

    lora_results.append({
        "project": "all",
        **calculate_metrics(global_pred_prob_lora,pred_label_lora, global_true_label),
        **effort_aware_metrics(test_expert_features,pd.read_csv(os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/lora_output.csv")))
    })
    base_results.append({
        "project": "all",
        **calculate_metrics(global_pred_prob_base,pred_label_base, global_true_label),
        **effort_aware_metrics(test_expert_features,pd.read_csv(os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/base_output.csv")))
    })
    outlier_results.append({
        "project": "all",
        **calculate_metrics(global_pred_prob_outlier,pred_label_outlier,global_true_label),
        **effort_aware_metrics(test_expert_features,pd.read_csv(os.path.join(RESULTS_DIR,f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/outlier_output.csv")))
    })    

    result_df = pd.DataFrame(lora_results)
    result_path = os.path.join(RESULTS_DIR, f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/lora_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"LoRA results saved to {result_path}")

    result_df = pd.DataFrame(base_results)
    result_path = os.path.join(RESULTS_DIR, f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/base_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Base results saved to {result_path}")

    result_df = pd.DataFrame(outlier_results)
    result_path = os.path.join(RESULTS_DIR, f"{args.cluster_model}/{args.n_cluster}/{args.pretrained_model}/outlier_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Base results saved to {result_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    ####################################### Load datasets ######################################################
    train_dataset_dict = load_project_datas(tokenizer, args, "train")
    ############################# 层次聚类 ####################################

    fcluster=HierarchicalCluster(args,n_cluster=args.n_cluster)
    fcluster.fit(train_dataset_dict)
    

    if args.base_model == "concat":
        mymodel = ConcatModel(model, config, tokenizer, args).to(device)
    elif args.base_model == "single":
        mymodel = SingleModel(model, config, tokenizer, args).to(device)
    elif args.base_model == "manual":
        mymodel = ManualModel(model, config, tokenizer, args).to(device)
    else:
        raise ValueError(f"Invalid base model: {args.base_model}")

    test_dataset_dict = load_project_datas(tokenizer, args, "test")
    valid_dataset_dict = load_project_datas(tokenizer, args, "eval")
    ############################# 层次聚类 ####################################
    test_dataset_dict = fcluster.splitDatasets(test_dataset_dict)
    valid_dataset_dict = fcluster.splitDatasets(valid_dataset_dict)

    model_name=f"{args.base_model}-{args.pretrained_model}-final.pt"
    save_dir=os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
    checkpoint = torch.load(os.path.join(save_dir,model_name),weights_only=True)
    mymodel.load_state_dict(checkpoint)

    ensure_directory_exists(os.path.join(RESULTS_DIR, f"{args.cluster_model}/{str(args.n_cluster)}"))

    weight_dict=weight_calculator(args,valid_dataset_dict,mymodel,fcluster)
    compare_test(args, test_dataset_dict, mymodel,fcluster,weight_dict)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_comparison(base_path, lora_path, outlier_path, metrics, output_dir):
    """
    对比可视化函数：生成三组数据的柱状对比图
    
    参数：
    base_path -- 基准数据路径
    lora_path -- LoRA数据路径
    outlier_path -- 异常数据路径
    metrics -- 需要对比的指标列表
    output_dir -- 输出目录路径
    """
    # 设置美观的样式
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '0.98'
    
    # 读取数据并设置索引
    base_df = pd.read_csv(base_path, index_col=0)
    lora_df = pd.read_csv(lora_path, index_col=0)
    outlier_df = pd.read_csv(outlier_path, index_col=0)

    # 验证数据一致性
    if not (base_df.index.equals(lora_df.index) and base_df.index.equals(outlier_df.index)):
        raise ValueError("输入数据的索引不一致！")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义更美观的配色
    colors = sns.color_palette("husl", 3)  # 使用seaborn的husl调色板

    # 遍历每个指标
    for metric in metrics:
        # 提取数据
        base = base_df[metric]
        lora = lora_df[metric]
        outlier = outlier_df[metric]

        # 配置绘图参数
        bar_width = 0.25
        index = np.arange(len(base))  # X轴位置
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制三组柱状图（使用新配色）
        bars_base = ax.bar(index - bar_width, base, bar_width, 
                          label='Base', color=colors[0])
        bars_lora = ax.bar(index, lora, bar_width, 
                          label='LoRA', color=colors[1])
        bars_outlier = ax.bar(index + bar_width, outlier, bar_width, 
                            label='Outlier', color=colors[2])

        # 设置坐标轴
        ax.set_xticks(index)
        ax.set_xticklabels(base.index, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Data Index', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'Comparison of {metric} Across Groups', fontsize=13, pad=20)
        ax.legend(frameon=True, shadow=True)

        # 添加数值标签（保留4位小数）
        for bars in [bars_base, bars_lora, bars_outlier]:
            ax.bar_label(bars, padding=3, fontsize=8, 
                        fmt='%.4f')  # 格式化为4位小数

        # 自动调整布局
        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"对比图表已保存至：{output_dir}")

if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for n_cluster in [4]:
        args.strategy="Isolation"
        args.n_cluster=n_cluster
        for base_model in ["concat"]:
            for pretrained in ["codebert","codet5", "graphcodebert", "unixcoder","plbart"]:
                print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
                args.cluster_model=f"project_final"
                args.base_model=base_model
                args.pretrained_model=pretrained
                with torch.no_grad():
                    main(args)
                torch.cuda.empty_cache()
                print("Test End")
                plot_comparison(
                    base_path=os.path.join(f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}", "base_results.csv"),
                    lora_path=os.path.join(f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}", "lora_results.csv"),
                    outlier_path=os.path.join(f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}", "outlier_results.csv"),
                    metrics=["f1", "gmean","auc","recall"],  # 需要对比的指标
                    output_dir=f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}"
                )
