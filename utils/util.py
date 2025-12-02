import argparse
import random
import numpy as np
import torch
import os
from peft import LoraConfig,TaskType
import csv
from transformers import (RobertaModel, RobertaTokenizer, RobertaConfig, T5ForConditionalGeneration, T5Config,T5EncoderModel,
                          PLBartTokenizer, PLBartForConditionalGeneration, PLBartConfig)
from sklearn.metrics import roc_auc_score, auc
import math
import transformers 

import csv
import os
from typing import List, Union

from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve,matthews_corrcoef


def parse_jit_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", nargs=2, type=str,default=["./dataset/changes_train.pkl","./dataset/features_train.pkl"])
    parser.add_argument("--eval_data_file", nargs=2, type=str,default=["./dataset/changes_valid.pkl","./dataset/features_valid.pkl"])
    parser.add_argument("--test_data_file", nargs=2, type=str,default=["./dataset/changes_test.pkl","./dataset/features_test.pkl"])
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--pretrained_model", type=str, default="codebert")
    parser.add_argument("--base_model", type=str, default="concat")
    parser.add_argument("--loss_fct", type=str, default="focal")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=2,
                        help='patience for early stop')
    parser.add_argument('--n_cluster', type=int, default=4)
    parser.add_argument('--cluster_model', type=str, default="kmean")
    parser.add_argument("--manual_feature_size", type=int, default=14)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh")

    parser.add_argument("--alpha",type=float,default=0.75)
    parser.add_argument("--gamma",type=float,default=2.0)
    parser.add_argument("--eps",type=float,default=1e-8)

    parser.add_argument("--strategy",type=str,default="Isolation",choices=["Isolation","Lof","svm","Mahalanobis","tri"],help="How do you check the outliers when use final model.")

    parser.add_argument("--max_msg_length", type=int, default=64)
    parser.add_argument("--max_input_tokens", type=int, default=512)

    parser.add_argument("--available_gpu", type=list, default=[0])
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--base_train", action='store_true',default=False)
    parser.add_argument("--lora_train", action='store_true',default=False)

    parser.add_argument("--load_model_path",type=str,default="./output/checkpoints/concat/cct5.bin")
    
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def build_model_tokenizer_config(args):
    model_classes = {
        "codebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/codebert-base"),
        "graphcodebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/graphcodebert-base"),
        "codet5": (T5EncoderModel, RobertaTokenizer, T5Config, "Salesforce/codet5-base"),
        "unixcoder": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/unixcoder-base"),
        "plbart": (PLBartForConditionalGeneration, PLBartTokenizer, PLBartConfig, "uclanlp/plbart-base"),
        "plbart-large": (PLBartForConditionalGeneration, PLBartTokenizer, PLBartConfig, "uclanlp/plbart-large")
    }

    model_class, tokenizer_class, config_class, actual_name = model_classes[args.pretrained_model]

    # load config.
    config = config_class.from_pretrained(actual_name)
    if args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
        config.hidden_size = args.hidden_size
    elif args.pretrained_model in ["codet5", "plbart", "plbart-large"]:
        config.d_model = args.hidden_size
    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.dropout
    # load tokenizer.
    tokenizer = tokenizer_class.from_pretrained(actual_name)
    special_tokens_dict = {"additional_special_tokens": ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    # load pretrained model.
    model = model_class.from_pretrained(actual_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, config

def get_peft_lora_config(args):
    target_modules = []

    if args.pretrained_model in ["codet5"]:
        target_modules += ["q", "v"]
    elif args.pretrained_model in ["plbart"]:
        target_modules += ["q_proj", "v_proj"]
    elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
        target_modules += ["query", "value"]
    elif args.pretrained_model in ["cct5"]:
        target_modules += ["q", "v","manual_dense","out_proj_new"]

    if args.lora_train:
        if args.base_model == "single":
            target_modules += ["ll_proj"]
        elif args.base_model == "concat":
            target_modules += ["manual_dense", "cat_proj"]
        elif args.base_model == "manual":
            target_modules += ["manual_dense", "ll_proj"]
    print(f"training layers: {target_modules}")
    # if args.pretrained_model in ["codet5"]:
    #     peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
    #                              lora_dropout=0.1, target_modules=target_modules)
    # elif args.pretrained_model in ["plbart"]:
    #     peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
    #                              lora_dropout=0.1, target_modules=target_modules)
    # elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
    #     peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
    #                              lora_dropout=0.1, target_modules=target_modules)
    peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                            lora_dropout=0.1, target_modules=target_modules)
    return peft_config


# Write training evaluation
def write_training_results(args,
                           epoch=None,
                           step=None,
                           results=None,
                           save_dir=None,
                           dataset_type="eval",
                           first_row=False,
                           log_file=None):
    if not first_row and results is None:
        raise ValueError("Results should be provided.")
    # write the result to the file.
    mode = "a" if not first_row else "w"
    if first_row:
        row = ["epoch", "step", "dataset", "accuracy", "recall","recall0", "precision", "f1", "gmean","mcc","eval_loss", "auc_score"]
    else:
        row = [epoch, step, dataset_type,
               results["eval_accuracy"],
               results["eval_recall"],
               results["eval_recall0"],
               results["eval_precision"],
               results["eval_f1"],
               results["eval_gmean"],
               results["eval_mcc"],
               results["eval_loss"],
               results["auc_score"]
               ]
    if log_file is not None:
        path=log_file
    else:
        # if there is no such directory, create it.
        training_result_dir = os.path.join(save_dir, "training_results")
        if not os.path.exists(training_result_dir):
            os.makedirs(training_result_dir)
        path = os.path.join(training_result_dir, f"training_base_model_{args.pretrained_model}.csv")
    with open(path, mode) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)


def ensure_directory_exists(directory_path):
    """

    :param directory_path: the path needed to be checked
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Create path: {directory_path}")
        except OSError as e:
            print(f"Failed Creating Path: {e}")
    else:
        print(f"The path is already exsisting: {directory_path}")


def convert_dtype_dataframe(df, feature_name):
    df = df.astype({i: 'float32' for i in feature_name})
    return df


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




def effort_aware_metrics(test_features, result_df):
    result_df = result_df.sort_values(by='commit_hash')
    gold=result_df["true_label"].values.tolist()
    prob=result_df["prob"].values.tolist()

    test_features = test_features[['commit_hash', 'la', 'ld']]
    test_features = test_features.sort_values(by='commit_hash')
    test_features['label'] = gold
    test_features = convert_dtype_dataframe(test_features, ['la', 'ld'])
    test_features['LOC'] = test_features['la'] + test_features['ld']

    loc_sum = sum(test_features['LOC'])
    test_features['defective_commit_prob'] = prob
    test_features['defect_density'] = test_features['defective_commit_prob'] / \
        test_features['LOC']  # predicted defect density
    test_features['actual_defect_density'] = test_features['label'] / \
        test_features['LOC']  # defect density

    result_df = test_features.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(
        by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(
        by='actual_defect_density', ascending=True)
    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()
    real_buggy_commits = result_df[result_df['label'] == 1]

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * loc_sum
    buggy_line_20_percent = result_df[result_df['cum_LOC']
                                      <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(
        buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(
        math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(
        buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                           real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                        real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df,
                                                                              real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(
            predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(
            actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(
            actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                     auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))


    result={
        "R20E":recall_20_percent_effort,
        "E20R":effort_at_20_percent_LOC_recall,
        "Popt":p_opt
    }
    return result

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * \
        result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC']
                                         <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(
        buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def write_test_results_to_csv(
    commit_ids: List[str],
    probs: List[float],
    prediction_labels: List[Union[int, str]],
    true_labels: List[Union[int, str]],
    cluster: str,
    filename: str = "results.csv"
):
    """
    将模型测试结果写入CSV文件
    
    参数:
        commit_ids: 样本编号列表
        probs: 预测概率列表
        prediction_labels: 预测结果列表
        true_labels: 真实标签列表
        cluster: 隶属簇名称
        filename: 输出文件名(默认为results.csv)
    """
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.isfile(filename)
    
    if not file_exists:
        with open(filename,'w',encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["commit_hash", "prob", "prediction_label", "true_label", "cluster"])
            
    # 打开文件(追加模式)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
                
        # 写入数据行
        for commit_id, prob, pred_label, true_label in zip(commit_ids, probs, prediction_labels, true_labels):
            writer.writerow([commit_id, prob, pred_label, true_label, cluster])