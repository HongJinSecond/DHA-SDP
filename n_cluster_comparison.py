"""
n_cluster_comparison.py

Compare validation performance across different n_cluster values [2, 4, 6, 8].
Supports three clustering modes: style (KMeans), project (Hierarchical), developer (Hierarchical).

For each n_cluster:
  1. Cluster training data -> train one LoRA per cluster
  2. Evaluate each cluster's best LoRA on its valid subset
  3. Aggregate predictions across all clusters -> compute overall metrics
  4. Record results

Output: CSV file with columns [cluster_type, encoder, n_cluster, metrics...]
        suitable for plotting n_cluster vs performance curves.

Usage:
    python n_cluster_comparison.py --do_train
    (Edit CLUSTER_TYPE, ENCODER, N_CLUSTERS_LIST below to configure)
"""

import torch
import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve, matthews_corrcoef
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, PeftModel
from torch.optim import AdamW

from utils.util import (parse_jit_args, set_seed, build_model_tokenizer_config,
                        ensure_directory_exists, get_peft_lora_config)
from utils.process_datasets import JITFineDataset, load_project_datas, load_developer_datas
from models.SingleModel import SingleModel
from models.ConcatModel import ConcatModel
from models.ManualModel import ManualModel
from config import *

from clusters.cluster_manager import ClusterManager
from clusters.feature_cluster import FeatureClusterProcessor
from clusters.HierarchicalCluster import HierarchicalCluster

import csv

logger = logging.getLogger(__name__)

# ===================== Configuration (edit these) =====================
CLUSTER_TYPES = ["style","project","developer"]
ENCODERS = ["codebert","codet5","graphcodebert","unixcoder","plbart"]           # "codebert" | "codet5" | "graphcodebert" | "unixcoder" | "plbart"
N_CLUSTERS_LIST = [2, 4, 6, 8]
# Set RESUME=True to skip clusters whose LoRA already exists on disk and
# load the saved adapter instead of retraining. Useful after an interrupted run.
RESUME = True
PLOT_METRIC = "gmean"  # metric to plot on y-axis (choose from METRIC_KEYS)
PLOT_FORMAT = "pdf"  # output format: "pdf" (vector) or "png" (raster)
# ======================================================================

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "gmean", "mcc", "auc"]
CSV_FIELDNAMES = ["cluster_type", "encoder", "n_cluster", "num_eval_samples"] + METRIC_KEYS


def calculate_metrics(pred_prob, true_label, threshold=0.5):
    pred_label = np.array([0 if x < threshold else 1 for x in pred_prob])
    true_label = np.array(true_label)
    fpr, tpr, _ = roc_curve(true_label, pred_prob)
    recall1 = recall_score(true_label, pred_label, average="binary", zero_division=0)
    recall0 = recall_score(true_label, pred_label, pos_label=0, average="binary", zero_division=0)
    return {
        "accuracy": (pred_label == true_label).mean(),
        "precision": precision_score(true_label, pred_label, average="binary", zero_division=0),
        "recall": recall1,
        "f1": f1_score(true_label, pred_label, average="binary", zero_division=0),
        "gmean": np.sqrt(recall0 * recall1) if recall0 > 0 and recall1 > 0 else 0.0,
        "mcc": matthews_corrcoef(true_label, pred_label),
        "auc": auc(fpr, tpr),
    }


def evaluate_model(args, eval_dataset, mymodel):
    """Evaluate model, return metrics dict with eval_loss (for early stopping)."""
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    pred_prob = []
    true_label = []
    total_loss = []
    mymodel.eval()

    for batch in eval_dataloader:
        _, input_ids, input_mask, manual_features, label = batch
        input_ids, input_mask, manual_features, label = (
            x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
        )
        with torch.no_grad():
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())
            total_loss.append(loss.cpu().numpy().mean().item())

    pred_prob = np.concatenate(pred_prob, 0)
    true_label = np.concatenate(true_label, 0)
    total_loss = np.mean(total_loss).item()

    result = calculate_metrics(pred_prob, true_label, args.threshold)
    result["eval_loss"] = total_loss
    return result


def get_predictions(args, eval_dataset, mymodel):
    """Evaluate model, return raw (pred_prob, true_label) for aggregation."""
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    pred_prob = []
    true_label = []
    mymodel.eval()

    for batch in eval_dataloader:
        _, input_ids, input_mask, manual_features, label = batch
        input_ids, input_mask, manual_features, label = (
            x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
        )
        with torch.no_grad():
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())

    return np.concatenate(pred_prob, 0), np.concatenate(true_label, 0)


def train_one_lora(args, train_dataset, eval_dataset, mymodel, style_name):
    """
    Train one LoRA adapter. After training, restore the best LoRA weights
    (by eval_loss) in memory so the caller can evaluate directly.
    Also saves the best LoRA to disk (same path structure as existing code).
    """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = max(1, len(train_dataloader) // 2)
    args.warmup_steps = 0

    optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    # Initial eval
    min_loss = None
    if eval_dataset is not None:
        initial = evaluate_model(args, eval_dataset, mymodel)
        min_loss = initial["eval_loss"]
    patience = 0
    mymodel.zero_grad()

    # Save paths (same structure as existing code)
    output_dir = os.path.join(LORA_DIR, args.cluster_model, str(args.n_cluster),
                              args.base_model, args.pretrained_model)
    ensure_directory_exists(output_dir)
    output_file = os.path.join(output_dir, style_name)
    ensure_directory_exists(output_file)

    # Save initial empty LoRA
    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
    model_to_save.save_pretrained(output_file)

    # Track best LoRA state in memory (only trainable params = LoRA weights)
    best_lora_state = None
    early_stop = False

    for idx in range(args.epochs):
        if early_stop:
            break
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"  Epoch {idx+1}")
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            _, input_ids, input_mask, manual_features, label = batch
            input_ids, input_mask, manual_features, label = (
                x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
            )
            mymodel.train()
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.warning(f"epoch {idx} step {step+1} loss {round(tr_loss/tr_num, 5)}")
                tr_loss = 0
                tr_num = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mymodel.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Eval + early stopping
            if (step + 1) % args.save_steps == 0 and eval_dataset is not None:
                results = evaluate_model(args, eval_dataset, mymodel)
                if results["eval_loss"] < min_loss:
                    patience = 0
                    min_loss = results["eval_loss"]
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    model_to_save.save_pretrained(output_file)
                    # Save best LoRA state in memory (only trainable params)
                    best_lora_state = {
                        name: param.data.clone()
                        for name, param in model_to_save.named_parameters()
                        if param.requires_grad
                    }
                    print(f"    Find new min loss {min_loss:.4f}, saved best LoRA for '{style_name}'")
                else:
                    patience += 1
                    if patience > args.patience * 2:
                        logger.info(f"patience exceeded, early stop for {style_name}")
                        early_stop = True
                        break

    # Restore best LoRA weights in memory
    if best_lora_state is not None:
        model_to_restore = mymodel.module if hasattr(mymodel, 'module') else mymodel
        for name, param in model_to_restore.named_parameters():
            if name in best_lora_state:
                param.data.copy_(best_lora_state[name])

    return mymodel


def build_base_model(args, device):
    """Build base model and load pretrained checkpoint."""
    model, tokenizer, config = build_model_tokenizer_config(args)
    if args.base_model == "concat":
        mymodel = ConcatModel(model, config, tokenizer, args).to(device)
    elif args.base_model == "single":
        mymodel = SingleModel(model, config, tokenizer, args).to(device)
    elif args.base_model == "manual":
        mymodel = ManualModel(model, config, tokenizer, args).to(device)
    else:
        raise ValueError(f"Invalid base model: {args.base_model}")

    base_model_name = f"{args.base_model}-{args.pretrained_model}-final.pt"
    base_model_dir = os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
    checkpoint_path = os.path.join(base_model_dir, base_model_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base model checkpoint not found: {checkpoint_path}")
    mymodel.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
    return mymodel, tokenizer, config


def setup_clustering(args, tokenizer, n_cluster, cluster_type):
    """
    Set up clustering and split train/eval data into cluster dicts.
    Returns: (train_cluster_dict, eval_cluster_dict)
    """
    if cluster_type == "style":
        args.cluster_model = "Kmean"
        cluster_model = FeatureClusterProcessor(args, n_clusters=n_cluster)
        cluster_manager = ClusterManager(args, cluster_model,
                                         os.path.join(LORA_DIR, f"{args.base_model}/{args.pretrained_model}"))
        train_dataset = JITFineDataset(tokenizer, args, "train")
        train_features = cluster_model.extract_features(train_dataset)
        cluster_manager.init_cluster_model(train_dataset, features=train_features)
        eval_dataset = JITFineDataset(tokenizer, args, "eval")
        train_cluster_dict = cluster_manager.cluster_model.clustered_data
        eval_cluster_dict = cluster_manager.split_datasets(eval_dataset)

    elif cluster_type == "project":
        args.cluster_model = "project_final"
        train_dataset_dict = load_project_datas(tokenizer, args, "train")
        fcluster = HierarchicalCluster(args, n_cluster=n_cluster)
        fcluster.fit(train_dataset_dict)
        train_cluster_dict = fcluster.splitDatasets(train_dataset_dict)
        eval_dataset_dict = load_project_datas(tokenizer, args, "eval")
        eval_cluster_dict = fcluster.splitDatasets(eval_dataset_dict)

    elif cluster_type == "developer":
        args.cluster_model = "developer_aware"
        train_dataset_dict = load_developer_datas(tokenizer, args, "train")
        fcluster = HierarchicalCluster(args, n_cluster=n_cluster)
        fcluster.fit(train_dataset_dict)
        train_cluster_dict = fcluster.splitDatasets(train_dataset_dict)
        eval_dataset_dict = load_developer_datas(tokenizer, args, "eval")
        eval_cluster_dict = fcluster.splitDatasets(eval_dataset_dict)

    else:
        raise ValueError(f"Invalid cluster_type: {cluster_type}")

    return train_cluster_dict, eval_cluster_dict


def run_one_n_cluster(args, device, tokenizer, n_cluster, cluster_type):
    """
    Run the full pipeline for one n_cluster value.
    Returns: dict of overall valid metrics (or None if failed).
    """
    args.n_cluster = n_cluster
    print(f"\n{'='*60}")
    print(f"  cluster_type={cluster_type}, n_cluster={n_cluster}, encoder={args.pretrained_model}")
    print(f"{'='*60}")

    train_cluster_dict, eval_cluster_dict = setup_clustering(args, tokenizer, n_cluster, cluster_type)

    peft_config = get_peft_lora_config(args)

    all_pred_prob = []
    all_true_label = []
    total_eval_samples = 0

    # Train + evaluate one LoRA per cluster
    for cluster_name, train_lora_dataset in train_cluster_dict.items():
        if cluster_name == "other":
            continue  # no LoRA for "other"

        if len(train_lora_dataset) == 0:
            print(f"  [Skip] Cluster '{cluster_name}' has 0 training samples")
            continue

        torch.cuda.empty_cache()
        print(f"\n  >> Training LoRA for cluster '{cluster_name}' "
              f"({len(train_lora_dataset)} train samples)")

        mymodel, _, _ = build_base_model(args, device)

        eval_lora_datasets = eval_cluster_dict.get(cluster_name)
        style_name = str(cluster_name)

        # Resume check: if LoRA adapter already exists on disk, load it instead of retraining
        lora_save_path = os.path.join(LORA_DIR, args.cluster_model, str(args.n_cluster),
                                      args.base_model, args.pretrained_model, style_name)
        lora_config_file = os.path.join(lora_save_path, "adapter_config.json")
        lora_done_marker = os.path.join(lora_save_path, "TRAIN_DONE")
        if RESUME and os.path.exists(lora_config_file):
            has_marker = os.path.exists(lora_done_marker)
            print(f"  [Resume] Cluster '{cluster_name}' LoRA found on disk"
                  f"{' (training completed)' if has_marker else ' (no TRAIN_DONE marker - verify training was complete)'}")
            print(f"           Loading from: {lora_save_path}")
            mymodel = PeftModel.from_pretrained(mymodel, lora_save_path)
            mymodel = mymodel.to(args.device)
        else:
            mymodel = get_peft_model(mymodel, peft_config)
            mymodel.print_trainable_parameters()
            # Train (restores best weights in memory)
            mymodel = train_one_lora(args, train_lora_dataset, eval_lora_datasets, mymodel, style_name)
            # Write completion marker so future RESUME runs can confidently skip this cluster
            with open(lora_done_marker, "w") as f:
                f.write("done\n")

        # Evaluate with best LoRA on this cluster's valid subset
        if eval_lora_datasets is not None and len(eval_lora_datasets) > 0:
            pred_prob, true_label = get_predictions(args, eval_lora_datasets, mymodel)
            all_pred_prob.append(pred_prob)
            all_true_label.append(true_label)
            total_eval_samples += len(true_label)
            print(f"     Eval on {len(true_label)} valid samples")
        else:
            print(f"     No valid data for cluster '{cluster_name}'")

        del mymodel
        torch.cuda.empty_cache()

    # Handle "other" cluster (eval data from unseen projects/developers) with base model
    if "other" in eval_cluster_dict and len(eval_cluster_dict["other"]) > 0:
        torch.cuda.empty_cache()
        print(f"\n  >> Evaluating 'other' cluster with base model "
              f"({len(eval_cluster_dict['other'])} samples)")
        mymodel, _, _ = build_base_model(args, device)
        pred_prob, true_label = get_predictions(args, eval_cluster_dict["other"], mymodel)
        all_pred_prob.append(pred_prob)
        all_true_label.append(true_label)
        total_eval_samples += len(true_label)
        del mymodel
        torch.cuda.empty_cache()

    # Aggregate
    if not all_pred_prob:
        print("  [Warning] No valid predictions collected!")
        return None

    all_pred_prob = np.concatenate(all_pred_prob, 0)
    all_true_label = np.concatenate(all_true_label, 0)

    metrics = calculate_metrics(all_pred_prob, all_true_label, args.threshold)
    metrics["n_cluster"] = n_cluster
    metrics["num_eval_samples"] = total_eval_samples

    print(f"\n  --- Overall valid metrics (n_cluster={n_cluster}) ---")
    for key in METRIC_KEYS:
        print(f"    {key:12s}: {metrics[key]:.4f}")
    print(f"    {'samples':12s}: {total_eval_samples}")

    return metrics


def print_comparison_table(results, cluster_type, encoder):
    """Print a formatted comparison table to console."""
    header = f"{'n_cluster':>10s} | " + " | ".join(f"{m:>10s}" for m in METRIC_KEYS)
    separator = "-" * len(header)

    print(f"\n{'='*70}")
    print(f"  Comparison Table: cluster_type={cluster_type}, encoder={encoder}")
    print(f"{'='*70}")
    print(header)
    print(separator)
    for r in results:
        row = f"{r['n_cluster']:>10d} | " + " | ".join(f"{r[m]:>10.4f}" for m in METRIC_KEYS)
        print(row)
    print(separator)


def _read_all_results(output_dir):
    """Read all *_comparison.csv files, return combined list of dicts."""
    import glob
    all_results = []
    for csv_path in sorted(glob.glob(os.path.join(output_dir, "*_comparison.csv"))):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = {
                    "cluster_type": row["cluster_type"],
                    "encoder": row["encoder"],
                    "n_cluster": int(row["n_cluster"]),
                }
                for m in METRIC_KEYS:
                    r[m] = float(row[m])
                all_results.append(r)
    return all_results


def plot_all_from_csv():
    """
    Read all result CSVs (including cct5) and generate two sets of plots:
      Mode 1 (by_encoder):  one plot per encoder, lines = cluster_types
      Mode 2 (by_cluster):  one plot per cluster_type, lines = encoders
    X = n_cluster, Y = PLOT_METRIC.
    """
    output_dir = os.path.join(RESULTS_DIR, "n_cluster_comparison")
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    all_results = _read_all_results(output_dir)
    if not all_results:
        print(f"No result CSVs found in {output_dir}")
        return

    encoders = sorted(set(r["encoder"] for r in all_results))
    cluster_types = sorted(set(r["cluster_type"] for r in all_results))
    metric = PLOT_METRIC

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    print(f"Plotting metric: {metric}")
    print(f"  Encoders found:      {encoders}")
    print(f"  Cluster types found: {cluster_types}")

    ENCODER_MAP = {"codebert": "CodeBERT", "codet5": "CodeT5", "graphcodebert": "GraphCodeBERT", "unixcoder": "UniXcoder", "plbart": "PLART","cct5":"CCT5"}

    # --- Mode 1: one plot per encoder, lines = cluster_types ---
    by_encoder_dir = os.path.join(output_dir, "by_encoder")
    ensure_directory_exists(by_encoder_dir)
    for encoder in encoders:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cluster_type in cluster_types:
            rows = sorted(
                [r for r in all_results
                 if r["encoder"] == encoder and r["cluster_type"] == cluster_type],
                key=lambda r: r["n_cluster"],
            )
            if not rows:
                continue
            ax.plot([r["n_cluster"] for r in rows],
                    [r[metric] for r in rows],
                    marker="o", linewidth=2, markersize=8, label=cluster_type)

        ax.set_xlabel("Cluster Number", fontsize=13)
        ax.set_ylabel("G-mean", fontsize=13)
        # ax.set_title(f"{metric} vs n_cluster  (encoder={encoder})", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(by_encoder_dir, f"{ENCODER_MAP[encoder]}_{metric}.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"  [by_encoder] {encoder}: {plot_path}")

    # --- Mode 2: one plot per cluster_type, lines = encoders ---
    by_cluster_dir = os.path.join(output_dir, "by_cluster")
    ensure_directory_exists(by_cluster_dir)
    for cluster_type in cluster_types:
        fig, ax = plt.subplots(figsize=(8, 5))
        for encoder in encoders:
            rows = sorted(
                [r for r in all_results
                 if r["cluster_type"] == cluster_type and r["encoder"] == encoder],
                key=lambda r: r["n_cluster"],
            )
            if not rows:
                continue
            ax.plot([r["n_cluster"] for r in rows],
                    [r[metric] for r in rows],
                    marker="o", linewidth=2, markersize=8, label=ENCODER_MAP[encoder] )

        ax.set_xlabel("Cluster Number", fontsize=13)
        ax.set_ylabel("G-mean", fontsize=13)
        # ax.set_title(f"{metric} vs n_cluster  (cluster_type={cluster_type})", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(by_cluster_dir, f"{cluster_type}_{metric}.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"  [by_cluster] {cluster_type}: {plot_path}")

    print(f"\nAll plots generated in: {output_dir}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # Apply configuration
    args.base_model = "concat"
    args.do_train = True

    # Build tokenizer once (reused across all runs)
    _, tokenizer, _ = build_model_tokenizer_config(args)

    output_dir = os.path.join(RESULTS_DIR, "n_cluster_comparison")
    ensure_directory_exists(output_dir)

    for cluster_type in CLUSTER_TYPES:
        print(f"\n{'#'*60}")
        print(f"#  Running cluster_type = {cluster_type}, encoder = {args.pretrained_model}")
        print(f"{'#'*60}")

        # Open CSV for this cluster_type + encoder (overwrite previous partial results)
        csv_path = os.path.join(output_dir, f"{cluster_type}_{args.pretrained_model}_comparison.csv")
        csv_f = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        csv_f.flush()

        results = []
        for n_cluster in N_CLUSTERS_LIST:
            set_seed(args)  # reset seed before each n_cluster for fair comparison
            metrics = run_one_n_cluster(args, device, tokenizer, n_cluster, cluster_type)
            if metrics is not None:
                results.append(metrics)
                # Write immediately to disk so results survive a crash
                row = {
                    "cluster_type": cluster_type,
                    "encoder": args.pretrained_model,
                    "n_cluster": metrics["n_cluster"],
                    "num_eval_samples": metrics["num_eval_samples"],
                }
                for m in METRIC_KEYS:
                    row[m] = round(metrics[m], 4)
                csv_writer.writerow(row)
                csv_f.flush()
            torch.cuda.empty_cache()

        csv_f.close()

        if results:
            print_comparison_table(results, cluster_type, args.pretrained_model)
            print(f"Results saved to: {csv_path}")
        else:
            print(f"No results collected for cluster_type={cluster_type}!")


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        plot_all_from_csv()
    else:
        args = parse_jit_args()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        for encoder in ENCODERS:
            args.pretrained_model = encoder
            main(args)
