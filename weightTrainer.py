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
from models.ConcatModel import ConcatModel
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









