# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
import os 
import argparse
import yaml
from termcolor import colored
import torch
import numpy as np
import random
from tqdm import tqdm
from models import AMMD, CombinedLoss
from datasets import get_dvlog_dataloader, kfold_get_dvlog_dataloader
from utils import plot_confusion_matrix, plot_confusion_matrix_mean
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold

def collate_fn(batch):
    # Assuming x is the data and y is the label
    data, labels = zip(*batch)
    # Example: pad data to max length in the batch
    max_length = max(d.shape[0] for d in data)
    padded_data = [np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant') for d in data]
    return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG :", 'green') + colored(msg, mcolor)) # type: ignore

# Seed 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2024)

CONFIG_PATH = "./config.yaml"

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model on the DVLOG dataset."
    )
    # arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m", "--model", type=str,
        choices=["AMMD"]
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument(
        "-sch", "--lr_scheduler", type=str,
        choices=["cos", "None",]
    )
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args

def val(
    net, val_loader, loss_fn, device, 
):
    """Test the model on the validation / test set."""
    net.eval()
    sample_count = 0
    running_loss = 0.
    running_msa_loss = 0.
    running_vtt_loss = 0.

    # Initialize variables for metrics
    TP, FP, TN, FN = 0, 0, 0, 0
    TP_msa, FP_msa, TN_msa, FN_msa = 0, 0, 0, 0
    TP_vtt, FP_vtt, TN_vtt, FN_vtt = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch"
        ) as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device).unsqueeze(1)

                # Pass to Model
                y_pred, msa_pred, vtt_pred = net(x)

                ## ammd loss (total loss)
                loss = loss_fn(y_pred, y.to(torch.float32), net)

                ## msa loss
                loss_msa = loss_fn(msa_pred, y.to(torch.float32), net)

                ## vtt loss  
                loss_vtt = loss_fn(vtt_pred, y.to(torch.float32), net) 

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                running_msa_loss += loss_msa.item() * x.shape[0]
                running_vtt_loss += loss_vtt.item() * x.shape[0]

                # binary classification with only one output neuron
                ## Total Preds
                pred = (y_pred > 0).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                ## MSA Preds
                pred_msa = (msa_pred > 0).int()
                TP_msa += torch.sum((pred_msa == 1) & (y == 1)).item()
                FP_msa += torch.sum((pred_msa == 1) & (y == 0)).item()
                TN_msa += torch.sum((pred_msa == 0) & (y == 0)).item()
                FN_msa += torch.sum((pred_msa == 0) & (y == 1)).item()

                ## VTTEncoder Preds
                pred_vtt = (vtt_pred > 0).int()
                TP_vtt += torch.sum((pred_vtt == 1) & (y == 1)).item()
                FP_vtt += torch.sum((pred_vtt == 1) & (y == 0)).item()
                TN_vtt += torch.sum((pred_vtt == 0) & (y == 0)).item()
                FN_vtt += torch.sum((pred_vtt == 0) & (y == 1)).item()

                # Calculate metrics for total loss
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                # Calculate metrics for MSA loss
                msa_precision = TP_msa / (TP_msa + FP_msa) if (TP_msa + FP_msa) > 0 else 0.0
                msa_recall = TP_msa / (TP_msa + FN_msa) if (TP_msa + FN_msa) > 0 else 0.0
                msa_f1_score = (
                    2 * (msa_precision * msa_recall) / (msa_precision + msa_recall) 
                    if (msa_precision + msa_recall) > 0 else 0.0
                )
                msa_accuracy = (
                    (TP_msa + TN_msa) / sample_count
                    if sample_count > 0 else 0.0
                )

                # Calculate metrics for VTTEncoder loss
                vtt_precision = TP_vtt / (TP_vtt + FP_vtt) if (TP_vtt + FP_vtt) > 0 else 0.0
                vtt_recall = TP_vtt / (TP_vtt + FN_vtt) if (TP_vtt + FN_vtt) > 0 else 0.0
                vtt_f1_score = (
                    2 * (vtt_precision * vtt_recall) / (vtt_precision + vtt_recall) 
                    if (vtt_precision + vtt_recall) > 0 else 0.0
                )
                vtt_accuracy = (
                    (TP_vtt + TN_vtt) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": running_loss / sample_count, 
                    "loss_msa": running_msa_loss / sample_count,
                    "loss_vtt": running_vtt_loss / sample_count,
                    "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                    "msa_acc": msa_accuracy,
                    "msa_precision": msa_precision, "msa_recall": msa_recall, "msa_f1": msa_f1_score,
                    "vtt_acc": vtt_accuracy,
                    "vtt_precision": vtt_precision, "vtt_recall": vtt_recall, "vtt_f1": vtt_f1_score,
                })

    return {
        "loss": running_loss / sample_count, 
        "loss_msa": running_msa_loss / sample_count,
        "loss_vtt": running_vtt_loss / sample_count,
        "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
        "msa_acc": msa_accuracy,
        "msa_precision": msa_precision, "msa_recall": msa_recall, "msa_f1": msa_f1_score,
        "vtt_acc": vtt_accuracy,
        "vtt_precision": vtt_precision, "vtt_recall": vtt_recall, "vtt_f1": vtt_f1_score,

        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TP_msa": TP_msa, "FP_msa": FP_msa, "TN_msa": TN_msa, "FN_msa": FN_msa,
        "TP_vtt": TP_vtt, "FP_vtt": FP_vtt, "TN_vtt": TN_vtt, "FN_vtt": FN_vtt,
    }

def main():
    args = parse_args()
    LOG_INFO(args)

    # Initialize K-Fold cross-validation
    train_loader = get_dvlog_dataloader(args.data_dir, "train", args.batch_size, args.train_gender)
    val_loader = get_dvlog_dataloader(args.data_dir, "valid", args.batch_size, args.test_gender)
    test_loader = get_dvlog_dataloader(args.data_dir, "test", args.batch_size, args.test_gender)

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(combined_dataset))

    ##-----------------------------------------------------------------
    # Initialize accumulators for TP, FP, TN, FN across all 10 folds
    TP_sum, FP_sum, TN_sum, FN_sum = 0, 0, 0, 0
    TP_msa_sum, FP_msa_sum, TN_msa_sum, FN_msa_sum = 0, 0, 0, 0
    TP_vtt_sum, FP_vtt_sum, TN_vtt_sum, FN_vtt_sum = 0, 0, 0, 0

    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        LOG_INFO(f"Fold {fold+1}/{args.num_folds}")

        train_subset = Subset(combined_dataset, train_indices.tolist())
        val_subset = Subset(combined_dataset, val_indices.tolist())

        train_loader_fold = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader_fold = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        # Construct the model
        net = AMMD(d=256, l=6)
        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        LOG_INFO(f"[{args.model}] Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}", "cyan")
        
        # Set other training components
        loss_fn = CombinedLoss(lambda_reg=1e-5, focal_weight=0.5, l2_weight=0.5)
        
        # Load the best model for testing
        net.load_state_dict(
            torch.load(os.path.join("./weights", f"best_model_wo_lrs_{fold}.pt"), map_location=args.device[0])
            # torch.load(os.path.join("./weights", f"best_model_{fold}.pt"), map_location=args.device[0])
        )
        test_results = val(net, val_loader_fold, loss_fn, args.device[0])

        ## calculate overall (avg) conf & vizualization -----------------------
        # Accumulate confusion matrix values for main model
        TP_sum += test_results["TP"]
        FP_sum += test_results["FP"]
        TN_sum += test_results["TN"]
        FN_sum += test_results["FN"]
        ## plots
        # plot_confusion_matrix(test_results["TP"], test_results["FP"], test_results["TN"], test_results["FN"], 
        #                       title=f"mainkfold_confusion_matrix_{fold}", filename=f"mainkfold_confusion_matrix_{fold}.png")

        # Accumulate confusion matrix values for MSA model
        TP_msa_sum += test_results["TP_msa"]
        FP_msa_sum += test_results["FP_msa"]
        TN_msa_sum += test_results["TN_msa"]
        FN_msa_sum += test_results["FN_msa"]
        ## plots
        # plot_confusion_matrix(test_results["TP_msa"], test_results["FP_msa"], test_results["TN_msa"], test_results["FN_msa"], 
        #                       title=f"mainkfold_msa_confusion_matrix_{fold}", filename=f"mainkfold_msa_confusion_matrix_{fold}.png")

        # Accumulate confusion matrix values for VTTEncoder model
        TP_vtt_sum += test_results["TP_vtt"]
        FP_vtt_sum += test_results["FP_vtt"]
        TN_vtt_sum += test_results["TN_vtt"]
        FN_vtt_sum += test_results["FN_vtt"]
        ## plots
        # plot_confusion_matrix(test_results["TP_vtt"], test_results["FP_vtt"], test_results["TN_vtt"], test_results["FN_vtt"], 
        #                       title=f"mainkfold_vtt_confusion_matrix_{fold}", filename=f"mainkfold_vtt_confusion_matrix_{fold}.png")
        ##-----------------------------------------------------------------

        fold_results.append(test_results)
        LOG_INFO(f"Fold {fold+1} test results: {fold_results[-1]}", mcolor='yellow')

    ##------------------- avg confusion matrix--------------------------
    # Compute the average confusion matrix for each model
    TP_avg, FP_avg, TN_avg, FN_avg = round(TP_sum / 10), round(FP_sum / 10), round(TN_sum / 10), round(FN_sum / 10)
    TP_msa_avg, FP_msa_avg, TN_msa_avg, FN_msa_avg = round(TP_msa_sum / 10), round(FP_msa_sum / 10), round(TN_msa_sum / 10), round(FN_msa_sum / 10)
    TP_vtt_avg, FP_vtt_avg, TN_vtt_avg, FN_vtt_avg = round(TP_vtt_sum / 10), round(FP_vtt_sum / 10), round(TN_vtt_sum / 10), round(FN_vtt_sum / 10)

    # Now you can use the previously provided code to plot the averaged confusion matrix
    plot_confusion_matrix(TP_avg, FP_avg, TN_avg, FN_avg, 
                          title="Main Model Averaged Confusion Matrix", filename="mainkfold_main_model_avg_confusion_matrix.png")
    plot_confusion_matrix(TP_msa_avg, FP_msa_avg, TN_msa_avg, FN_msa_avg, 
                          title="MSA Model Averaged Confusion Matrix", filename="mainkfold_msa_avg_confusion_matrix.png")
    plot_confusion_matrix(TP_vtt_avg, FP_vtt_avg, TN_vtt_avg, FN_vtt_avg, 
                          title="VTTEncoder Model Averaged Confusion Matrix", filename="mainkfold_vtt_avg_confusion_matrix.png")

    ##----------------------------------------

    print("============---------------================")
    LOG_INFO("Overall Folds Results for the test sets")
    LOG_INFO(f"All Results: {fold_results}")

    # Aggregate results across folds
    avg_results = {
        "acc": np.mean([fr["acc"] for fr in fold_results]),
        "precision": np.mean([fr["precision"] for fr in fold_results]),
        "recall": np.mean([fr["recall"] for fr in fold_results]),
        "f1": np.mean([fr["f1"] for fr in fold_results]),

        "loss": np.mean([fr["loss"] for fr in fold_results]),
        "loss_msa": np.mean([fr.get("loss_msa", 0) for fr in fold_results]),
        "loss_vtt": np.mean([fr.get("loss_vtt", 0) for fr in fold_results]),

        "msa_acc": np.mean([fr.get("msa_acc", 0) for fr in fold_results]),
        "msa_precision": np.mean([fr.get("msa_precision", 0) for fr in fold_results]),
        "msa_recall": np.mean([fr.get("msa_recall", 0) for fr in fold_results]),
        "msa_f1": np.mean([fr.get("msa_f1", 0) for fr in fold_results]),

        "vtt_acc": np.mean([fr.get("vtt_acc", 0) for fr in fold_results]),
        "vtt_precision": np.mean([fr.get("vtt_precision", 0) for fr in fold_results]),
        "vtt_recall": np.mean([fr.get("vtt_recall", 0) for fr in fold_results]),
        "vtt_f1": np.mean([fr.get("vtt_f1", 0) for fr in fold_results]),

        "TP": round(np.mean([fr.get("TP", 0) for fr in fold_results])),
        "FP": round(np.mean([fr.get("FP", 0) for fr in fold_results])),
        "TN": round(np.mean([fr.get("TN", 0) for fr in fold_results])),
        "FN": round(np.mean([fr.get("FN", 0) for fr in fold_results])),

        "TP_msa": round(np.mean([fr.get("TP_msa", 0) for fr in fold_results])),
        "FP_msa": round(np.mean([fr.get("FP_msa", 0) for fr in fold_results])),
        "TN_msa": round(np.mean([fr.get("TN_msa", 0) for fr in fold_results])),
        "FN_msa": round(np.mean([fr.get("FN_msa", 0) for fr in fold_results])),

        "TP_vtt": round(np.mean([fr.get("TP_vtt", 0) for fr in fold_results])),
        "FP_vtt": round(np.mean([fr.get("FP_vtt", 0) for fr in fold_results])),
        "TN_vtt": round(np.mean([fr.get("TN_vtt", 0) for fr in fold_results])),
        "FN_vtt": round(np.mean([fr.get("FN_vtt", 0) for fr in fold_results])),
    }

    ## Now you can use the previously provided code to plot the averaged confusion matrix
    # plot_confusion_matrix(avg_results['TP'], avg_results['FP'], avg_results['TN'], avg_results['FN'], 
    #                       title="AMMD Confusion Matrix", filename="mainkfold_ammd_confusion_matrix.png")
    # plot_confusion_matrix(avg_results['TP_msa'], avg_results['FP_msa'], avg_results['TN_msa'], avg_results['FN_msa'],
    #                       title="MSA Confusion Matrix", filename="mainkfold_msa_confusion_matrix.png")
    # plot_confusion_matrix(avg_results['TP_vtt'], avg_results['FP_vtt'], avg_results['TN_vtt'], avg_results['FN_vtt'],
    #                       title="VTTEncoder Confusion Matrix", filename="mainkfold_vtt_confusion_matrix.png")


    ## Now you can use mean confusion matrix
    plot_confusion_matrix_mean(avg_results['TP'], avg_results['FP'], avg_results['TN'], avg_results['FN'], 
                          title="Mean AMMD Confusion Matrix", filename="mean_mainkfold_ammd_confusion_matrix.png")
    plot_confusion_matrix_mean(avg_results['TP_msa'], avg_results['FP_msa'], avg_results['TN_msa'], avg_results['FN_msa'],
                          title="Mean MSA Confusion Matrix", filename="mean_mainkfold_msa_confusion_matrix.png")
    plot_confusion_matrix_mean(avg_results['TP_vtt'], avg_results['FP_vtt'], avg_results['TN_vtt'], avg_results['FN_vtt'],
                          title="Mean VTTEncoder Confusion Matrix", filename="mean_mainkfold_vtt_confusion_matrix.png")

    ##-----------------------------------------------------------------

    LOG_INFO(f"Average cross-validated results: {avg_results}", mcolor='green')

if __name__ == "__main__":
    main()


