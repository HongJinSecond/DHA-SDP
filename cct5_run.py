#This file is for model cct5, contains the training and test process.

import logging

logger = logging.getLogger(__name__)

from utils.util import parse_jit_args
from utils.process_datasets import JITFineDataset
from torch.utils.data import RandomSampler,DataLoader
from models.CCT5 import build_or_load_gen_model

import peft
import torch

import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve,matthews_corrcoef
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW

from models.FinalModel import FinalModel
from models.FinalModel import FinalModelCCT5

from utils.util import parse_jit_args,ensure_directory_exists,write_training_results,get_peft_lora_config,set_seed,plot_comparison,write_test_results_to_csv
from utils.process_datasets import load_project_datas,load_developer_datas
from config import *

from clusters.HierarchicalCluster import HierarchicalCluster
from clusters.cluster_manager import ClusterManager
from clusters.feature_cluster import FeatureClusterProcessor
from utils.util import effort_aware_metrics
import pandas as pd
from weightTrainer import weight_calculator

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




def train(args, train_dataset, eval_dataset, mymodel,Style_name=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0

    optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)
    print("Base performance")
    if eval_dataset!=None:
        initial = evaluate(args, eval_dataset, mymodel)
        best_gmean = initial["eval_gmean"]
    patience = 0
    mymodel.zero_grad()

    #######################################Save Path###############################################
    # the runtime model file path:
    output_dir = os.path.join(LORA_DIR, args.cluster_model)
    output_dir = os.path.join(output_dir, str(args.n_cluster))
    output_dir = os.path.join(output_dir,args.base_model)
    output_dir = os.path.join(output_dir,args.pretrained_model)
    #######################################Run Time Log Information File Path:TODO: Can Be removed###########################################
    log_dir = os.path.join(RUNTIME_DIR, "lora")
    log_dir = os.path.join(log_dir, args.cluster_model)
    log_dir = os.path.join(log_dir,str(args.n_cluster))
    log_dir = os.path.join(log_dir,args.base_model)
    log_dir = os.path.join(log_dir,args.pretrained_model)
    ensure_directory_exists(log_dir)
    ensure_directory_exists(output_dir)
    log_file= os.path.join(log_dir,f"{Style_name}.csv")
    # Lora Name
    output_file = os.path.join(output_dir, Style_name)

    ###################################TODO Removed when polished############################################
    write_training_results(args, first_row=True,log_file=log_file)
    #################################### First save, empty Lora #########################################################
    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
    model_to_save.save_pretrained(output_file)
    #################### First Save ; If one Lora is worse than base, we will still save it, just an empty Lora
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader),desc=f"Epoch {idx + 1}")
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            _, input_ids, input_mask, manual_features, label = batch
            input_ids, input_mask, manual_features, label = (
                x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
            )  
            mymodel.train()
            logits,loss = mymodel(
                cls=True,
                input_ids=input_ids,
                manual_feature=manual_features,
                labels=label,
                attention_mask=input_mask
            )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
  
            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.warning(f"epoch {idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            # truncate the gradient, used to prevent exploding gradient.
            torch.nn.utils.clip_grad_norm_(mymodel.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # save model after save_steps.
            if (step + 1) % args.save_steps == 0:
                if eval_dataset is not None:
                    results = evaluate(args, eval_dataset, mymodel)
                    ##################################TODO removed after experiment######################
                    # Write the runtime information
                    write_training_results(args, epoch=idx, step=step+1,
                                            results=results,
                                            log_file=log_file)
                    #######################################################################################表现性能肯定是先下降再上升的，所以如果以最开始最优的表现作为早停是会出错的
                    if results["eval_gmean"] > best_gmean:
                        patience = 0
                        best_gmean = results["eval_gmean"]
                    #################################Save the best lora model####################################
                        model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                        print(f"Find new best gmean {best_gmean}, save status!")
                        print(f"saved in {output_file}")
                        model_to_save.save_pretrained(output_file)
                    else:
                        patience += 1
                        if patience > args.patience * 5:
                            logger.info('patience greater than {}, early stop!'.format(args.patience))
                            return output_file
                else:
                    #################################### Dont have suitable evaluate dataset, just save it #################################
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    model_to_save.save_pretrained(output_file)

    return output_file


def compare_test(args, test_dataset_dict, base_model,cluster_manager,weight_dict:dict):
    #load model
    base_model.eval()
    finalModel = FinalModelCCT5(args, base_model)
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

        base_case=False

        if project_name == "other":
            base_case=True


        # print(type(project_name))
        print(f"Project: {project_name}: {len(test_cluster_datasets)} samples.")
        test_sampler = SequentialSampler(test_cluster_datasets)
        test_dataloader = DataLoader(test_cluster_datasets, sampler=test_sampler, batch_size=args.batch_size)
        # load lora 
        weight=weight_dict[project_name]
        if project_name != "other":
            finalModel.load_lora(
                os.path.join(LORA_DIR, f"{args.cluster_model}/{args.n_cluster}/{args.base_model}/{args.pretrained_model}/{project_name}"),
                lora_name=str(project_name))
            finalModel.change_loras(str(project_name))

        for batch in test_dataloader:
            if project_name=="other":
                outlier_mask=None
            else:
                outlier_mask=cluster_manager.checkOutliers(batch[3].numpy(),lora_key=project_name,strategy=args.strategy)
                outlier_mask=torch.tensor(outlier_mask).to(args.device)
                if outlier_mask.sum()==0:
                    print("No outlier.")
                    outlier_mask=None
            commit_hash, input_ids, input_mask, manual_features, label = batch
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
                # print(f"get weight: {weight}")
                prob, loss = finalModel(input_ids, input_mask, manual_features, label,use_base=base_case,outlier_mask=outlier_mask,weight=weight)
                global_pred_prob_outlier.append(prob.detach().cpu().numpy())
                local_pred_prob_outlier.append(prob.detach().cpu().numpy())
                ###############Run for lora without mask
                prob, loss = finalModel(input_ids, input_mask, manual_features, label,use_base=base_case)
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

def evaluate(args, eval_dataset, mymodel):
    """
    Evaluate the runtime model on the eval_dataset.
    :param args:
    :param eval_dataset:
    :param mymodel:
    :return:
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    pred_prob = []
    true_label = []
    total_loss=[]
    pred_label=[]
    mymodel.eval()

    for batch in eval_dataloader:
        _, input_ids, input_mask, manual_features, label = batch
        input_ids, input_mask, manual_features, label = (
            x.to(args.device) for x in (input_ids, input_mask, manual_features, label)
        )
        with torch.no_grad():
            logits,loss = mymodel(
                cls=True,
                input_ids=input_ids,
                manual_feature=manual_features,
                labels=label,
                attention_mask=input_mask
            )

            prediction = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            prob = torch.nn.functional.softmax(
                    logits, dim=1).data.cpu().numpy()[:, 1].tolist()            

            pred_prob.append(prob)
            true_label.append(label.cpu().numpy())
            total_loss.append(loss.detach().cpu().numpy().mean().item())
            pred_label.extend(prediction.tolist())

    pred_prob = np.concatenate(pred_prob, 0)
    true_label = np.concatenate(true_label, 0)
    
    # best_threshold = args.threshold
    total_loss=np.mean(total_loss).item()
    accuracy = (pred_label == true_label).mean()
    precision = precision_score(true_label, pred_label, average="binary")
    recall = recall_score(true_label, pred_label, average="binary")
    f1 = f1_score(true_label, pred_label, average="binary")
    recall0 = recall_score(true_label, pred_label, pos_label=0, average="binary")
    gmean = np.sqrt(recall0 * recall)
    mcc = matthews_corrcoef(true_label,pred_label)
    fpr, tpr, thres = roc_curve(true_label, pred_prob)
    auc_score = auc(fpr, tpr)

    result = {
        "eval_accuracy": accuracy,
        "eval_recall": recall,
        "eval_recall0": recall0,
        "eval_precision": precision,
        "eval_f1": f1,
        "eval_gmean": gmean,
        "eval_mcc":mcc,
        "auc_score": auc_score,
        "eval_loss":total_loss,

    }

    logger.info("***** Eval result *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    mymodel.train()
    return result



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    config, model, tokenizer = build_or_load_gen_model(args, load_model=False)

    if args.cluster_model=="project_final":
        ####################################### Load datasets ######################################################
        ######################### Project Aware ##################################
        train_dataset_dict = load_project_datas(tokenizer, args, "train")
        fcluster=HierarchicalCluster(args,n_cluster=args.n_cluster)
        fcluster.fit(train_dataset_dict)

        train_dataset_dict = fcluster.splitDatasets(train_dataset_dict)
        test_dataset_dict = load_project_datas(tokenizer, args, "test")
        valid_dataset_dict = load_project_datas(tokenizer, args, "eval")
        ############################# 层次聚类 ####################################
        test_dataset_dict = fcluster.splitDatasets(test_dataset_dict)
        valid_dataset_dict = fcluster.splitDatasets(valid_dataset_dict)

    elif args.cluster_model=="developer_aware":
        ############################# Developer aware
        train_dataset_dict = load_developer_datas(tokenizer, args, "train")
        fcluster=HierarchicalCluster(args,n_cluster=args.n_cluster)
        fcluster.fit(train_dataset_dict)

        train_dataset_dict = fcluster.splitDatasets(train_dataset_dict)
        test_dataset_dict = load_developer_datas(tokenizer, args, "test")
        valid_dataset_dict = load_developer_datas(tokenizer, args, "eval")
        ############################# 层次聚类 ####################################
        test_dataset_dict = fcluster.splitDatasets(test_dataset_dict)
        valid_dataset_dict = fcluster.splitDatasets(valid_dataset_dict)

    else:
        ####################### Simple for Kmean
        cluster_model = FeatureClusterProcessor(args,n_clusters=args.n_cluster)
        fcluster = ClusterManager(args, cluster_model,
                                     os.path.join(LORA_DIR, f"{args.base_model}/{args.pretrained_model}"))

        ####################################### Load datasets ######################################################
        train_dataset = JITFineDataset(tokenizer, args, "train")
        train_features = cluster_model.extract_features(train_dataset)
        fcluster.init_cluster_model(train_dataset,train_features)
        train_dataset_dict = fcluster.cluster_model.clustered_data
        test_dataset = JITFineDataset(tokenizer, args, "test")
        valid_dataset = JITFineDataset(tokenizer, args, "eval")
        test_dataset_dict=fcluster.split_datasets(test_dataset)
        valid_dataset_dict=fcluster.split_datasets(valid_dataset)
    

    if args.do_train:
        print("do train")
        peft_config = get_peft_lora_config(args)

        ######################## Split the datasets for many loras #####################################
        for project_name,train_lora_dataset in train_dataset_dict.items():
            print(type(project_name))
            if str(project_name)=="other":
                continue
            torch.cuda.empty_cache()
            config, model, tokenizer = build_or_load_gen_model(args, load_model=False)
            model.init_MF_classifier()
            model.load_state_dict(torch.load("./output/checkpoints/concat/cct5.bin", map_location="cpu"))
            model.eval()
            model.zero_grad()
            mymodel=get_peft_model(model,peft_config)
            mymodel.print_trainable_parameters()
            eval_lora_datasets = valid_dataset_dict.get(project_name)
            train(args, train_lora_dataset, eval_lora_datasets, mymodel,Style_name=str(project_name))

    if args.do_test:
        print("do test")
        model.init_MF_classifier()
        model.load_state_dict(torch.load("./output/checkpoints/concat/cct5.bin", map_location="cpu"))
        model.eval()
        model.zero_grad()
        ensure_directory_exists(os.path.join(RESULTS_DIR, f"{args.cluster_model}/{str(args.n_cluster)}"))
        weight_dict=weight_calculator(args,valid_dataset_dict,model,fcluster)
        with torch.no_grad():
            compare_test(args, test_dataset_dict, model,fcluster,weight_dict)


if __name__=="__main__":
    args=parse_jit_args()
    args.device="cuda"
    args.add_lang_ids=False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for base_model in ["concat"]:
        for pretrained in ["cct5"]:
            for n_cluster in [4]:
                for cluster_model in ["project_final","developer_aware","Kmean"]:
                    args.strategy="Isolation"
                    args.n_cluster=n_cluster
                    print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
                    args.base_model=base_model
                    args.cluster_model=cluster_model
                    args.pretrained_model=pretrained
                    main(args)
                    torch.cuda.empty_cache()
                    print("Test Start")
                    plot_comparison(
                        base_path=os.path.join(f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}", "base_results.csv"),
                        lora_path=os.path.join(f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}", "lora_results.csv"),
                        outlier_path=os.path.join(f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}", "outlier_results.csv"),
                        metrics=["f1", "gmean","auc","recall"],  # 需要对比的指标
                        output_dir=f"result/{args.cluster_model}/{str(args.n_cluster)}/{args.pretrained_model}"
                    )