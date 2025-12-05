'''
This file is for Base Model training, you can train your own base model or just use our pre-trained Base Model.
'''


import torch
import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, auc, roc_curve,matthews_corrcoef
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from utils.util import parse_jit_args, set_seed, build_model_tokenizer_config,ensure_directory_exists,write_training_results
from utils.process_datasets import JITFineDataset
from models.ConcatModel import ConcatModel
from config import *

logger = logging.getLogger(__name__)


def train(args, train_dataset, eval_dataset, mymodel):
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

    min_loss = 20
    patience = 0
    mymodel.zero_grad()

    #######################################Save Path###############################################
    # the runtime model file path:
    checkpoint_prefix = "checkpoint-best-f1"
    output_dir = os.path.join(RUNTIME_DIR, f"{checkpoint_prefix}")
    output_dir = os.path.join(output_dir,args.base_model)
    # make sure the directory exists
    ensure_directory_exists(output_dir)
    output_file = os.path.join(output_dir, f"{args.pretrained_model}_based.pt")

    ################################### TODO Removed when polished ############################################
    # write the runtime information
    write_training_results(args, first_row=True,save_dir=output_dir)
    ##########################################################################################################

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader),desc=f"Epoch {idx + 1}")
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
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
                results = evaluate(args, eval_dataset, mymodel)
                ##################################TODO removed after experiment######################
                # Write the runtime information
                write_training_results(args, epoch=idx, step=step+1,
                                        results=results,
                                        save_dir=output_dir)
                #######################################################################################
                if results["eval_loss"] < min_loss:
                    patience = 0
                    min_loss = results["eval_loss"]
                #################################Save the best model####################################
                    model_to_save = mymodel.module if hasattr(mymodel, 'module') else mymodel
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                    print(f"Find new min loss {min_loss}, save status!")
                    torch.save(save_content, output_file)

                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return output_file
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
    mymodel.eval()
    total_loss=[]
    for batch in eval_dataloader:
        input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
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
        "eval_precision": precision,
        "eval_f1": f1,
        "eval_gmean": gmean,
        "eval_mcc":mcc,
        "eval_recall0": recall0,
        "auc_score": auc_score,
        "eval_loss":total_loss,
    }

    logger.info("***** Eval result *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, test_dataset, mymodel):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    pred_prob = []
    true_label = []
    mymodel.eval()

    for batch in test_dataloader:
        input_ids, input_mask, manual_features, label = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob, loss = mymodel(input_ids, input_mask, manual_features, label)
            pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())

    pred_prob = np.concatenate(pred_prob, 0)
    true_label = np.concatenate(true_label, 0)
    best_threshold = args.threshold

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
        "test_accuracy": accuracy,
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1,
        "test_gmean": gmean,
        "test_mcc":mcc,
        "auc_score": auc_score
    }


    logger.info("***** Test result *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    set_seed(args)

    if args.do_train:
        model, tokenizer, config = build_model_tokenizer_config(args)
        if args.pretrained_model in ["codet5"]:
            peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["q", "v"])
        elif args.pretrained_model in ["plbart"]:
            peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
        elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
            peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["query", "value"])
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(f"trainable layer:{peft_config}")

        mymodel = ConcatModel(model, config, tokenizer, args).to(device)

        train_dataset = JITFineDataset(tokenizer, args, "train")
        eval_dataset = JITFineDataset(tokenizer, args, "eval")
        the_best_model_file=train(args, train_dataset, eval_dataset, mymodel)
        # Change the parameters to the best

        mymodel.load_state_dict(torch.load(the_best_model_file)["model_state_dict"],strict=True)
        model_name=f"{args.base_model}-{args.pretrained_model}-final.pt"
        # Merge the overall lora into baseline models.
        save_dir=os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
        mymodel.encoder =  mymodel.encoder.merge_and_unload()
        # Save the state dictionary.
        ensure_directory_exists(save_dir)
        torch.save(mymodel.state_dict(), os.path.join(save_dir,model_name))

    ### This part can test the performance of Base Model ###
    if args.do_test:
        model, tokenizer, config = build_model_tokenizer_config(args)
        mymodel = ConcatModel(model, config, tokenizer, args).to(device)

        test_dataset = JITFineDataset(tokenizer, args, "test")
        model_name=f"{args.base_model}-{args.pretrained_model}-final.pt"
        save_dir=os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
        checkpoint = torch.load(os.path.join(save_dir,model_name),weights_only=True)
        mymodel.load_state_dict(checkpoint)
        test(args, test_dataset, mymodel)


if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # "concat" means we both use expert feature and semantic feature
    for base_model in ["concat"]:
        for pretrained in ["codet5"]:
            # You can add more pre-trained models like ["codebert","codet5", "graphcodebert", "unixcoder","plbart"]
            print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
            args.base_model=base_model
            args.pretrained_model=pretrained
            main(args)
            torch.cuda.empty_cache()
