"""
统计三种聚类策略下，test数据集中每个簇的离群点占比，并保存为CSV和饼状图(PDF)。

步骤：
1. 对 Kmeans / developer_aware / project_final 三种策略，加载对应的 cluster_manager 和 test 数据
2. 遍历每个簇，调用 checkOutliers 统计离群点数量
3. 将结果保存到 CSV
4. 绘制饼状图保存为 PDF
"""

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler

from utils.util import parse_jit_args, set_seed, build_model_tokenizer_config
from utils.process_datasets import load_developer_datas, load_project_datas, JITFineDataset
from models.ConcatModel import ConcatModel
from config import *

from clusters.HierarchicalCluster import HierarchicalCluster
from clusters.cluster_manager import ClusterManager
from clusters.feature_cluster import FeatureClusterProcessor


def extract_expert_features(data, cluster_target=3):
    """从 JITFine Dataset 中提取 manual features"""
    return np.stack([item[cluster_target] for item in data], axis=0)


def collect_outlier_stats_for_kmeans(args, tokenizer):
    """Kmeans (Data-Driven) 策略"""
    args.cluster_model = "Kmeans"
    args.n_cluster = 4
    args.strategy = "Isolation"

    model, _, config = build_model_tokenizer_config(args)
    cluster_model = FeatureClusterProcessor(args, n_clusters=args.n_cluster)
    cluster_manager = ClusterManager(args, cluster_model,
                                     os.path.join(LORA_DIR, f"{args.base_model}/{args.pretrained_model}"))

    train_dataset = JITFineDataset(tokenizer, args, "train")
    train_features = cluster_model.extract_features(train_dataset)
    cluster_manager.init_cluster_model(train_dataset, train_features)

    test_dataset = JITFineDataset(tokenizer, args, "test")
    test_dataset_split = cluster_manager.split_datasets(test_dataset)

    results = []
    for lora_name, dataset in test_dataset_split.items():
        lora_name = int(lora_name)
        all_masks = []
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)
        for batch in dataloader:
            mask = cluster_manager.checkOutliers(batch[3].numpy(), lora_key=lora_name, strategy=args.strategy)
            all_masks.append(mask)
        all_masks = np.concatenate(all_masks)
        total = len(all_masks)
        outlier_count = int(np.sum(all_masks))
        normal_count = total - outlier_count

        label = f"Cluster {lora_name}" if lora_name != -1 else "Outlier"
        results.append({
            "strategy": "Kmeans",
            "cluster": label,
            "total": total,
            "outlier_count": outlier_count,
            "normal_count": normal_count,
            "outlier_ratio": outlier_count / total if total > 0 else 0
        })
    return results


def collect_outlier_stats_for_developer(args, tokenizer):
    """Developer-Aware 策略"""
    args.cluster_model = "developer_aware"
    args.n_cluster = 4
    args.strategy = "Isolation"

    train_dataset_dict = load_developer_datas(tokenizer, args, "train")
    fcluster = HierarchicalCluster(args, n_cluster=args.n_cluster)
    fcluster.fit(train_dataset_dict)

    test_dataset_dict = load_developer_datas(tokenizer, args, "test")
    test_dataset_dict = fcluster.splitDatasets(test_dataset_dict)

    results = []
    for developer_name, dataset in test_dataset_dict.items():
        if developer_name == "other":
            # other 簇不做离群点检测（原始代码中 outlier_mask=None）
            total = len(dataset)
            results.append({
                "strategy": "Developer Aware",
                "cluster": "Other",
                "total": total,
                "outlier_count": 0,
                "normal_count": total,
                "outlier_ratio": 0.0
            })
            continue

        all_masks = []
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)
        for batch in dataloader:
            mask = fcluster.checkOutliers(batch[3].numpy(), lora_key=developer_name, strategy=args.strategy)
            all_masks.append(mask)
        all_masks = np.concatenate(all_masks)
        total = len(all_masks)
        outlier_count = int(np.sum(all_masks))
        normal_count = total - outlier_count

        results.append({
            "strategy": "Developer Aware",
            "cluster": f"Cluster {developer_name}",
            "total": total,
            "outlier_count": outlier_count,
            "normal_count": normal_count,
            "outlier_ratio": outlier_count / total if total > 0 else 0
        })
    return results


def collect_outlier_stats_for_project(args, tokenizer):
    """Project-Aware 策略"""
    args.cluster_model = "project_final"
    args.n_cluster = 4
    args.strategy = "Isolation"

    train_dataset_dict = load_project_datas(tokenizer, args, "train")
    fcluster = HierarchicalCluster(args, n_cluster=args.n_cluster)
    fcluster.fit(train_dataset_dict)

    test_dataset_dict = load_project_datas(tokenizer, args, "test")
    test_dataset_dict = fcluster.splitDatasets(test_dataset_dict)

    results = []
    for project_name, dataset in test_dataset_dict.items():
        if project_name == "other":
            total = len(dataset)
            results.append({
                "strategy": "Project Aware",
                "cluster": "Other",
                "total": total,
                "outlier_count": 0,
                "normal_count": total,
                "outlier_ratio": 0.0
            })
            continue

        all_masks = []
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)
        for batch in dataloader:
            mask = fcluster.checkOutliers(batch[3].numpy(), lora_key=project_name, strategy=args.strategy)
            all_masks.append(mask)
        all_masks = np.concatenate(all_masks)
        total = len(all_masks)
        outlier_count = int(np.sum(all_masks))
        normal_count = total - outlier_count

        results.append({
            "strategy": "Project Aware",
            "cluster": f"Cluster {project_name}",
            "total": total,
            "outlier_count": outlier_count,
            "normal_count": normal_count,
            "outlier_ratio": outlier_count / total if total > 0 else 0
        })
    return results


def plot_outlier_pie_charts(df, output_dir="pic"):
    """为每种策略绘制离群点占比饼状图"""
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#FF6B6B', '#4ECDC4']  # 离群点/正常点

    for strategy in df["strategy"].unique():
        sub_df = df[df["strategy"] == strategy]
        n_clusters = len(sub_df)

        fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))
        if n_clusters == 1:
            axes = [axes]

        for i, (_, row) in enumerate(sub_df.iterrows()):
            ax = axes[i]
            outlier_count = int(row["outlier_count"])
            normal_count = int(row["normal_count"])
            sizes = [outlier_count, normal_count]

            if all(s == 0 for s in sizes):
                sizes = [1e-9, 1 - 1e-9]

            def make_autopct(outlier_count, normal_count):
                def my_autopct(pct):
                    total = outlier_count + normal_count
                    val = int(round(pct * total / 100.0))
                    return f'{val}\n({pct:.1f}%)'
                return my_autopct

            wedges, texts, autotexts = ax.pie(
                sizes,
                autopct=make_autopct(outlier_count, normal_count),
                colors=colors,
                startangle=90,
                textprops={'fontsize': 14}
            )

            ax.set_title(f'{row["cluster"]}\n(N={row["total"]})', fontweight='bold', fontsize=16)
            ax.legend(wedges, ['Outlier', 'Normal'], loc='upper left', fontsize=13)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(18)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{strategy.replace(' ', '_')}_outlier_ratio.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"饼图已保存至: {save_path}")


def plot_from_csv(csv_path="pic/outlier_stats.csv", output_dir="pic"):
    """从CSV文件读取离群点统计数据并绘制饼状图，跳过 Other 簇"""
    df = pd.read_csv(csv_path)
    # 过滤掉 Other 簇
    df = df[df["cluster"] != "Other"].copy()
    plot_outlier_pie_charts(df, output_dir)


def main():
    args = parse_jit_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])
    set_seed(args)

    _, tokenizer, _ = build_model_tokenizer_config(args)

    all_results = []

    print("=" * 60)
    print("正在统计 Kmeans 离群点...")
    print("=" * 60)
    all_results.extend(collect_outlier_stats_for_kmeans(args, tokenizer))

    print("=" * 60)
    print("正在统计 Developer Aware 离群点...")
    print("=" * 60)
    all_results.extend(collect_outlier_stats_for_developer(args, tokenizer))

    print("=" * 60)
    print("正在统计 Project Aware 离群点...")
    print("=" * 60)
    all_results.extend(collect_outlier_stats_for_project(args, tokenizer))

    # 保存 CSV
    df = pd.DataFrame(all_results)
    os.makedirs("pic", exist_ok=True)
    csv_path = os.path.join("pic", "outlier_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n离群点统计已保存至: {csv_path}")
    print(df.to_string(index=False))

    # 绘制饼状图
    plot_outlier_pie_charts(df, output_dir="pic")


if __name__ == "__main__":
    # main()
    plot_from_csv()
