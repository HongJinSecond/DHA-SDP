from collections import defaultdict

import torch
import logging
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve, matthews_corrcoef
from torch.utils.data import DataLoader, SequentialSampler,RandomSampler
import pandas as pd
from models.FinalModel import FinalModelWeight,FinalModelCCT5
from utils.util import parse_jit_args, set_seed, build_model_tokenizer_config, get_peft_lora_config, \
    ensure_directory_exists
from utils.process_datasets import load_project_datas
from models.SingleModel import SingleModel
from models.ConcatModel import ConcatModel
from models.ManualModel import ManualModel
from config import *
import json
from clusters.HierarchicalCluster import HierarchicalCluster
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

def weight_calculator(args,test_dataset_dict,base_model,cluster_manager):
    """
    使用最小二乘法计算解析解
    :param args:
    :param test_dataset_dict:
    :param base_model:
    :param cluster_manager:
    :return:
    """
    base_model.eval()
    if args.pretrained_model=="cct5":
        finalModel=FinalModelCCT5(args,base_model)
    else:
        finalModel = FinalModelWeight(args, base_model)
    finalModel.eval()
    weight_dict=defaultdict(lambda: 0.2)

    for style_name, test_cluster_datasets in test_dataset_dict.items():
        if str(style_name)=="other" or style_name==-1:
            continue
        print(f"Project: {style_name}: {len(test_cluster_datasets)} samples.")

        test_sampler = RandomSampler(test_cluster_datasets)
        test_dataloader = DataLoader(test_cluster_datasets, sampler=test_sampler, batch_size=args.batch_size)
        
        # load lora
        finalModel.load_lora(
            os.path.join(LORA_DIR,
                         f"{args.cluster_model}/{args.n_cluster}/{args.base_model}/{args.pretrained_model}/{style_name}"),
            lora_name=str(style_name))
        finalModel.change_loras(str(style_name))

        P1=[]
        P2=[]
        true_label=[]

        for batch in test_dataloader:
            # check outlier samples
            outlier_mask = cluster_manager.checkOutliers(batch[3].numpy(), lora_key=style_name,
                                                         strategy=args.strategy)
            outlier_mask = torch.tensor(outlier_mask).to(args.device)
            if outlier_mask.sum()==0:
                continue
            # only care about outliers!!!
            _, input_ids, input_mask, manual_features, label = batch
            input_ids, input_mask, manual_features, label = (
                x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
            )
            with torch.no_grad():
                true_label.append(label[outlier_mask].detach().cpu().numpy())
                ###############RUN for outlier strategy, weight is pre-trained
                finalModel.change_loras(str(style_name))
                prob_base, loss = finalModel(input_ids[outlier_mask], input_mask[outlier_mask], manual_features[outlier_mask], label[outlier_mask], use_base=True)
                prob_lora, loss2 = finalModel(input_ids[outlier_mask], input_mask[outlier_mask], manual_features[outlier_mask], label[outlier_mask], use_base=False)

                P1.append(prob_base.detach().cpu().numpy())
                P2.append(prob_lora.detach().cpu().numpy())

        P1=np.concatenate(P1, axis=0)
        P2=np.concatenate(P2, axis=0)
        true_label=np.concatenate(true_label, axis=0)

        def objective(w):
        
            prob = (1 - w) * P1 + w * P2
        
            # 确保输入形状一致
            prob = prob.reshape(-1,1)
            target = true_label.astype(float).reshape(-1,1)  # 转换为float以匹配where的条件
        
            # 数值稳定性处理，防止log(0)
            prob = np.clip(prob, 1e-8, 1 - 1e-8)
        
            # 计算p_t：当target=1时取prob，否则取1-prob
            p_t = np.where(target == 1, prob, 1 - prob)
        
            # 计算交叉熵损失项
            ce_loss = -np.log(p_t)
        
            # 计算alpha因子：类别1使用alpha，类别0使用1-alpha
            alpha_t = np.where(target == 1, 0.75, 1 - 0.75)
        
            # 计算调制因子和总损失
            focal_loss = alpha_t * np.pow(1 - p_t, 2) * ce_loss
        
            return np.sum(focal_loss)
        
        # 设置约束 w ∈ [0,1]
        bounds = [(0, 1)]  # w必须在[0,1]范围内
        
        # 初始猜测
        initial_guess = 0.5
        
        # 求解优化问题
        result = minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 40
                }
            )                                          
        if result.success:
            w_optimal = result.x[0]
            print(f"最优权重 w = {w_optimal:.4f}")
        
            weight_dict[style_name] = w_optimal
        else:
            print("优化失败:", result.message)


    return weight_dict

def calculate_metrics(pred_label, true_label):
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
    }

    return metrics

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
    ####################################### 层次聚类 #########################################
    fcluster = HierarchicalCluster(args)
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
    valid_dataset_dict = fcluster.splitDatasets(valid_dataset_dict)
    test_dataset_dict = fcluster.splitDatasets(test_dataset_dict)


    model_name = f"{args.base_model}-{args.pretrained_model}-final.pt"
    save_dir = os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
    checkpoint = torch.load(os.path.join(save_dir, model_name), weights_only=True)
    mymodel.load_state_dict(checkpoint)

    ensure_directory_exists(os.path.join(RESULTS_DIR, f"{args.cluster_model}/{args.strategy}"))
    compare_test(args,test_dataset_dict,mymodel,fcluster)
    # weight_dict=weight_trainer(args, valid_dataset_dict, mymodel, fcluster)
    # weight_dict=weight_calculator(args, valid_dataset_dict, mymodel, fcluster)
    # print(weight_dict)
    # with open("weight_dict.txt", "w") as f:
    #     f.write(weight_dict)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison(paths, labels, metrics, output_dir):
    """
    对比可视化函数：生成多组数据的柱状对比图

    参数：
    paths -- CSV文件路径列表
    labels -- 每组数据的标签列表
    metrics -- 需要对比的指标列表
    output_dir -- 输出目录路径
    """
    # 验证输入
    if len(paths) != len(labels):
        raise ValueError("路径数量与标签数量不一致！")
    if not paths:
        raise ValueError("至少需要一个数据文件路径！")

    # 设置美观的样式
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '0.98'

    # 读取所有数据
    dfs = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)

    # 验证所有数据的索引一致性
    base_index = dfs[0].index
    for i, df in enumerate(dfs[1:], start=1):
        if not base_index.equals(df.index):
            raise ValueError(f"第{i + 1}个数据({paths[i]})与第一个数据的索引不一致！")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 动态生成配色方案
    num_groups = len(dfs)
    colors = sns.color_palette("husl", num_groups)  # 根据组数生成颜色

    # 遍历每个指标
    for metric in metrics:
        # 配置绘图参数
        bar_width = 0.8 / num_groups  # 动态调整柱子宽度
        index = np.arange(len(base_index))  # X轴位置
        fig, ax = plt.subplots(figsize=(12 + num_groups, 6))  # 动态调整画布宽度

        # 绘制每组数据的柱状图
        all_bars = []
        for i, df in enumerate(dfs):
            data = df[metric]
            # 计算柱子位置（居中分布）
            position = index + i * bar_width - (num_groups - 1) * bar_width / 2
            bars = ax.bar(position, data, bar_width,
                          label=labels[i], color=colors[i])
            all_bars.append(bars)

            # 添加数值标签
            ax.bar_label(bars, padding=3, fontsize=8, fmt='%.4f')

        # 设置坐标轴
        ax.set_xticks(index)
        ax.set_xticklabels(base_index, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Data Index', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'Comparison of {metric} Across Groups', fontsize=13, pad=20)
        ax.legend(frameon=True, shadow=True, loc='best')

        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

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

    print("Test Start")
    for strategy in ["Isolation"]:
        args.strategy = strategy
        for base_model in ["concat"]:
            for pretrained in ["codebert"]:
                print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
                args.cluster_model = f"project_f_1e-4"
                args.base_model = base_model
                args.pretrained_model = pretrained
                with torch.no_grad():
                    main(args)
                torch.cuda.empty_cache()
        plot_comparison(
            paths=["result/project_f_1e-4/Isolation/base_results.csv",
                "result/project_f_1e-4/Isolation/lora_results.csv",
                   "result/project_f_1e-4/Isolation/human_results.csv",
                   "result/project_f_1e-4/Isolation/weight_results.csv",
                   "result/project_f_1e-4/Isolation/weight_results2.csv",
                   "result/project_f_1e-4/Isolation/weight_results3.csv"
                   ],

            labels=["Base","Lora","weight0.2","pre-trained","optimization","Logistic"],
            metrics=["f1", "gmean","mcc","recall"],  # 需要对比的指标
            output_dir=f"result/{args.cluster_model}/{args.strategy}"
        )
        print("Test End")



