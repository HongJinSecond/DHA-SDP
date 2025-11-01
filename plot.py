import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from config import *


def plot_base_train_performance(base_models:list[str],pre_traineds:list[str],metric:str):
    labels=[]
    paths=[]
    for i in range(len(base_models)):
        labels.append(f"{base_models[i]}_{pre_traineds[i]}")
        paths.append(os.path.join(RUNTIME_DIR,f"checkpoint-best-f1/{base_models[i]}/training_results/training_base_model_{pre_traineds[i]}.csv"))
    if len(paths) == 1:
        plot_sole_models(paths[0],metric,labels[0])
    else:
        plot_multi_models(paths,metric,labels)

def plot_lora_train_performance(base_model:str,pre_trained:str,metric="f1"):
    labels=[]
    paths=[]
    paths.append(os.path.join(RUNTIME_DIR,f"checkpoint-best-f1/{base_model}/training_results/training_base_model_{pre_trained}.csv"))
    labels.append("base")
    for i in range(4):
        labels.append(str(i))
        paths.append(os.path.join(RUNTIME_DIR,f"lora/kmean4/{base_model}/{pre_trained}/{i}.csv"))
    plot_multi_models(paths,metric,labels)




def plot_sole_models(path,metric,labels=None):
    df = pd.read_csv(path)
    data = df[metric].values
    steps = range(len(data))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, data, 'b-', linewidth=2)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} Progression\n{os.path.basename(path)}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_multi_models(paths,metric,labels):
    """
    绘制单个或多个模型的指定指标变化曲线。

    参数:
        paths (str 或 list): CSV文件的路径或路径列表。
        metric (str): 要绘制的指标名称，如'eval_accuracy'。
    """
    # 统一处理路径为列表形式

    all_data = []
    max_steps = 0
    labels = labels

    # 读取所有数据并确定最大步数
    for path in paths:
        df = pd.read_csv(path)
        data = df[metric].values
        all_data.append(data)
        max_steps = max(max_steps, len(data))

    # 数据填充处理
    padded_data = []
    for data in all_data:
        if len(data) < max_steps:
            fill_value = max(data)  # 使用该模型该指标的最大值填充
            padded = np.concatenate([data, [fill_value] * (max_steps - len(data))])
        else:
            padded = data
        padded_data.append(padded)

    # 绘制对比曲线
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))  # 使用不同颜色区分模型

    for i, (data, label) in enumerate(zip(padded_data, labels)):
        plt.plot(range(max_steps), data,
                 color=colors[i],
                 linewidth=1.5,
                 alpha=0.8,
                 label=label)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'Model Comparison: {metric} Progression', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(base_path, lora_path, metrics, output_dir):
    """对比可视化函数"""
    # 读取数据
    base_df = pd.read_csv(base_path)
    lora_df = pd.read_csv(lora_path)

    # 合并数据
    merged_df = pd.merge(
        base_df,
        lora_df,
        on="style",
        suffixes=("_base", "_lora")
    )

    # 绘制多个指标
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # 生成对比柱状图
        x = np.arange(len(merged_df))
        width = 0.35

        # 绘制柱状图并保存对象
        plt.bar(x - width / 2, merged_df[f"{metric}_base"],
                width, label="Base Model")
        plt.bar(x + width / 2, merged_df[f"{metric}_lora"],
                width, label="LoRA Model")

        # 添加数据标签
        ax = plt.gca()
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + 0.001,  # 加小偏移量防止重叠
                    f'{height:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=8)

        # 图表装饰
        plt.title(f"{metric.upper()} Comparison")
        plt.ylabel(metric)
        plt.xticks(x, merged_df["style"], rotation=45)
        plt.legend()
        plt.tight_layout()

        # 保存结果
        save_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(save_path)
        plt.close()
        plt.show()
        print(f"Comparison plot saved to {save_path}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_multiple_csv_comparison(csv_paths, metrics, labels=None, output_dir=None):
    """
    对比多个CSV文件最后一行在指定指标上的差异

    参数:
        csv_paths: list, CSV文件路径列表
        metrics: list, 需要对比的指标名称列表(最多4个)
        labels: list, 每个CSV文件的标签(可选)
        output_dir: str, 输出目录(可选)
    """
    if len(metrics) > 4:
        raise ValueError("最多支持对比4个指标")

    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(csv_paths))]

    # 读取所有CSV文件的最后一行
    last_rows = []
    for path in csv_paths:
        df = pd.read_csv(path)
        last_rows.append(df.iloc[-1])  # 获取最后一行

    # 提取指定指标数据
    data = {metric: [] for metric in metrics}
    for row in last_rows:
        for metric in metrics:
            data[metric].append(row[metric])

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    if len(metrics) < 4:
        # 如果指标少于4个，隐藏多余的子图
        for i in range(len(metrics), 4):
            fig.delaxes(axes.flatten()[i])

    # 绘制每个指标的柱状图
    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        x = np.arange(len(labels))
        width = 0.8  # 柱宽

        bars = ax.bar(x, data[metric], width, color=plt.cm.tab20.colors[:len(labels)])

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    height + 0.01 * max(data[metric]),  # 动态偏移
                    f'{height:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=10)

        # 图表装饰
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存或显示结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "metrics_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"对比图已保存至: {save_path}")
    else:
        plt.show()


from clusters.HierarchicalCluster import HierarchicalCluster


def extract_expert_features(data,cluster_target=2):
    """extract expert features from JITFine Dataset"""
    return np.stack([item[cluster_target] for item in data], axis=0)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
    
import matplotlib.pyplot as plt
import numpy as np

def outlierPieFigure(cluster_manager: 'HierarchicalCluster', 
                    target_data_dict: dict, 
                    fig_name: str,
                    strategy: str = "Isolation"):
    """
    绘制多个数据集的离群值比例饼图，所有子图共享一个图例
    
    参数：
    cluster_manager - 聚类管理器对象
    target_data_dict - 目标数据字典，格式：{key: data}
    fig_name - 输出文件名
    strategy - 异常检测策略（默认Isolation）
    """
    
    # 初始化画布
    width = len(target_data_dict)
    plt.figure(figsize=(4 * width, 3))
    fig, axs = plt.subplots(1, width, figsize=(4 * width, 3))
    
    # 颜色配置（离群值用警示色，正常值用中性色）
    colors = ['#FF6B6B', '#4ECDC4']  # 珊瑚红 & 软青色
    handles = None  # 用于收集图例句柄

    for j, (key, value) in enumerate(target_data_dict.items()):
        ax = axs[j] if width > 1 else axs  # 处理单个子图情况
        
        # 隐藏坐标轴
        ax.axis('equal')  # 保证饼图为正圆
        ax.axis('off')    # 隐藏坐标轴

        # 计算离群值比例
        mask = cluster_manager.checkOutliers(
            extract_expert_features(value), 
            str(key), 
            strategy
        )
        outliers = np.sum(mask) / len(mask) if len(mask) > 0 else 0
        normals = 1 - outliers

        # 绘制饼图
        sizes = [outliers, normals]
        if all(s == 0 for s in sizes):  # 处理全0异常情况
            sizes = [1e-9, 1 - 1e-9]    # 避免除以0错误
            
        wedges, _, autotexts = ax.pie(
            sizes,
            colors=colors,
            autopct=lambda p: f'{p:.1f}%',
            textprops={'fontsize': 20, 'color': 'gray'}
        )

        # 收集第一个子图的图例句柄
        if j == 0 and handles is None:
            handles = wedges

        # 添加标题
        ax.set_title(f'{key}\n(N={len(value)})', fontsize=10, pad=10)

    # 添加共享图例
    fig.legend(
        handles=handles,
        labels=['Outliers', 'Normals'],
        title='Label',
        title_fontproperties={'size': 15}
    )

    fig.suptitle(fig_name)
    # 调整布局并保存
    plt.tight_layout(w_pad=2)
    # plt.subplots_adjust(right=0.8)
    # plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    # plt.close()

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


def cluster_distribution():
    # 数据
    '''
    Project Aware
    data = {
        "train": {
            "1": 8698,
            "2": 2004,
            "3": 3508,
            "4": 2164
        },
        "valid": {
            "1": 2902,
            "2": 671,
            "3": 1170,
            "4": 722
        },
        "test": {
            "1": 2913,
            "2": 673,
            "3": 1171,
            "4": 723
        }
    }
    '''


    '''
    Kmean 
    data = {
        "train": {
            "3": 2324,
            "2": 2824,
            "0": 10816,
            "1": 410
        },
        "valid": {
            "1": 147,
            "2": 2113,
            "3": 609,
            "0": 2596
        },
        "test": {
            "0": 2713,
            "2": 1728,
            "3": 813,
            "1": 226
        }
    }
    '''


    '''
    Developer Aware
    data = {
        "train": {
            "1": 2418,
            "2": 814,
            "3": 10269,
            "4": 136
        },
        "valid": {
            "1": 589,
            "3": 2943,
            "other": 513
        },
        "test": {
            "1": 637,
            "3": 1400,
            "other": 3443
        }
    }
    '''

    data = {
        "train": {
            "3": 2324,
            "2": 2824,
            "0": 10816,
            "1": 410
        },
        "valid": {
            "1": 147,
            "2": 2113,
            "3": 609,
            "0": 2596
        },
        "test": {
            "0": 2713,
            "2": 1728,
            "3": 813,
            "1": 226
        }
    }


    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 颜色设置
    colors = ['#ff9999', '#66b3ff', "#7dcf7d", '#ffcc99', '#ff99cc', '#c2c2f0']

    # 绘制每个数据集的饼图
    for i, (dataset_name, dataset_data) in enumerate(data.items()):
        labels = list(dataset_data.keys())
        values = list(dataset_data.values())
        
        # 计算百分比
        total = sum(values)
        percentages = [f'{(v/total)*100:.1f}%' for v in values]
        
        # 自定义autopct函数，显示数值和百分比
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f'{val}\n({pct:.1f}%)'
            return my_autopct
        
        # 绘制饼图
        wedges, texts, autotexts = axes[i].pie(
            values, 
            labels=labels, 
            autopct=make_autopct(values),
            colors=colors[:len(labels)],
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        # 设置标题
        axes[i].set_title(f'{dataset_name.upper()} Dataset\n(Total: {total})', fontweight='bold', fontsize=12)
        
        # 美化文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

    # 添加总标题
    plt.suptitle('Kmean Distribution Across Datasets', fontsize=16, fontweight='bold')

    # 调整布局
    plt.tight_layout()
    plt.show()

    # 打印详细数据统计
    print("详细数据统计:")
    print("=" * 50)
    for dataset_name, dataset_data in data.items():
        total = sum(dataset_data.values())
        print(f"\n{dataset_name.upper()} Dataset (Total: {total}):")
        print("-" * 30)
        for label, value in dataset_data.items():
            percentage = (value / total) * 100
            print(f"  {label}: {value} ({percentage:.1f}%)")


# 使用示例
if __name__ == "__main__":
    # 1. 定义CSV文件路径
    # csv_files = [
    #     "result/project_f_1e-4/Isolation/base_results.csv",
    #     "result/project_f_1e-4/Isolation/lora_results.csv",
    #     "result/project_f_1e-4/Isolation/outlier_results.csv",
    #     "result/project_f_1e-4/Isolation_0.01/outlier_results.csv",
    #     "result/project_f_1e-4/Isolation_0.05/outlier_results.csv",
    #     "result/project_f_1e-4/Isolation_0.1/outlier_results.csv",
    # ]
    #
    # # 2. 定义要对比的指标(最多4个)
    # target_metrics = ["gmean","f1","recall","mcc"]  # 替换为您的实际指标名
    #
    # # 3. (可选)定义每个模型的标签
    # model_labels = ["Base", "lora","Iso0.2","Iso0.01","Iso0.05","Iso0.1"]  # 示例标签
    #
    # # 4. 调用函数
    # plot_multiple_csv_comparison(
    #     csv_paths=csv_files,
    #     metrics=target_metrics,
    #     labels=model_labels,
    #     # output_dir="runtime"  # 设为None则显示而不保存
    # )

    # plot_comparison(
    #     paths=["result/project_f_1e-4/Isolation/base_results.csv",
    #         "result/project_f_1e-4/Isolation/lora_results.csv",
    #         "result/project_f_1e-4/Isolation_auto/outlier_results.csv",
    #         "result/project_f_1e-4/Isolation_0.01/outlier_results.csv",
    #         "result/project_f_1e-4/Isolation_0.05/outlier_results.csv",
    #         "result/project_f_1e-4/Isolation_0.1/outlier_results.csv"],
    #     labels=["Base", "lora","Iso_auto","Iso0.01","Iso0.05","Iso0.1"],
    #     metrics=["f1", "gmean", "mcc", "recall"],  # 需要对比的指标
    #     output_dir=f"result/project_f_1e-4/Isolation_compare"
    # )
    cluster_distribution()
