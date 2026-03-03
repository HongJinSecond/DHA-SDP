import torch
import os
import dill
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve,matthews_corrcoef
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW

from models.FinalModel import FinalModel
from utils.util import parse_jit_args, set_seed, build_model_tokenizer_config,ensure_directory_exists,write_training_results,get_peft_lora_config
from utils.process_datasets import JITFineDataset
from models.ConcatModel import ConcatModel
from config import *

from clusters.cluster_manager import ClusterManager
from clusters.feature_cluster import FeatureClusterProcessor


'''
This file is for the strategy -- Data Driven. 

Run this script and train Loras for Base Model based on Data Driven Aware (Kmeans in our work).

'''

logger = logging.getLogger(__name__)


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

    if eval_dataset!=None:
        initial = evaluate(args, eval_dataset, mymodel)
        min_loss = initial["eval_loss"]
    patience = 0
    mymodel.zero_grad()

    #######################################Save Path###############################################
    # the runtime model file path:
    output_dir = os.path.join(LORA_DIR, args.cluster_model)
    output_dir = os.path.join(output_dir, str(args.n_cluster))
    output_dir = os.path.join(output_dir,args.base_model)
    output_dir = os.path.join(output_dir,args.pretrained_model)
    ensure_directory_exists(output_dir)
    # Lora Name
    output_file = os.path.join(output_dir, Style_name)

    # write the runtime information
    ensure_directory_exists(output_file)
    #################################### First save, empty Lora #########################################################
    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
    model_to_save.save_pretrained(output_file)
    ##########################################################################################################

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
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            if args.n_gpu > 1:
                loss = loss.mean()

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
                    #######################################################################################表现性能肯定是先下降再上升的，所以如果以最开始最优的表现作为早停是会出错的
                    if results["eval_loss"] < min_loss:
                        patience = 0
                        min_loss = results["eval_loss"]
                    #################################Save the best lora model####################################
                        model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                        print(f"Find new min loss {min_loss}, save status!")
                        model_to_save.save_pretrained(output_file)
                    else:
                        patience += 1
                        if patience > args.patience * 4:
                            logger.info('patience greater than {}, early stop!'.format(args.patience))
                            return output_file
                else:
                    #################################### Dont have suitable evaluate dataset, just save it #################################
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    model_to_save.save_pretrained(output_file)

    return output_file

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
    best_threshold = args.threshold
    total_loss=np.mean(total_loss).item()
    pred_label = [0 if x < best_threshold else 1 for x in pred_prob]
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

    return result



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    ###################################### Choose Cluster Model #################################################
    cluster_model = FeatureClusterProcessor(args,n_clusters=args.n_cluster)
    cluster_manager = ClusterManager(args, cluster_model,
                                     os.path.join(LORA_DIR, f"{args.base_model}/{args.pretrained_model}"))

    ####################################### Load datasets ######################################################
    train_dataset = JITFineDataset(tokenizer, args, "train")
    train_features = cluster_model.extract_features(train_dataset)

    ######################## Clean and Filter Data ################################################
    # train_dataset,train_features = cluster_model.clean_datasets(train_dataset,train_features)
    cluster_manager.init_cluster_model(train_dataset,features=train_features)

    if args.do_train:
        peft_config = get_peft_lora_config(args)
        eval_dataset = JITFineDataset(tokenizer, args, "eval")
        ######################## Split the datasets for many loras #####################################
        train_cluster_dict = cluster_manager.cluster_model.clustered_data
        eval_cluster_dict = cluster_manager.split_datasets(eval_dataset)
        for label,train_lora_dataset in train_cluster_dict.items():
            torch.cuda.empty_cache()
            ################################# The Base Model should be reload for training each Lora #################################
            model, tokenizer, config = build_model_tokenizer_config(args)

            if args.base_model == "concat":
                mymodel = ConcatModel(model, config, tokenizer, args).to(device)
            else:
                raise ValueError(f"Invalid base model: {args.base_model}")

            ###################################### Load Base Model ####################################################
            base_model_name = f"{args.base_model}-{args.pretrained_model}-final.pt"
            base_model_dir = os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
            mymodel.load_state_dict(torch.load(os.path.join(base_model_dir, base_model_name),weights_only=True),strict=True)
            # Make sure
            mymodel = get_peft_model(mymodel, peft_config)
            mymodel.print_trainable_parameters()
            eval_lora_datasets = eval_cluster_dict.get(label)
            # Train Target Lora
            train(args, train_lora_dataset, eval_lora_datasets, mymodel,Style_name=str(label))

        # cluster_manager.cluster_model.save(f"{OUTPUT_DIR}/lora/{args.cluster_model}/mycluster_model.bin") # Save cluster model


if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for base_model in ["concat"]:
        # PLMs
        for pretrained in ["codebert","codet5", "graphcodebert", "unixcoder","plbart"]:
            print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
            for i in [4]:
                args.base_model=base_model
                args.n_cluster=i
                args.cluster_model=f"Kmeans"
                args.pretrained_model=pretrained
                main(args)
                torch.cuda.empty_cache()
