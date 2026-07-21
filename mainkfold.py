"""K-fold training and evaluation entry point for AMDD on D-Vlog."""

import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import yaml
from datetime import datetime
from termcolor import colored
import torch
import numpy as np
import random
from tqdm import tqdm
from models import AMDD, CombinedLoss
from datasets import get_dvlog_dataloader
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold


class EarlyStopping:
    def __init__(self, patience=15, delta=0, verbose=False, save_path=None):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"Validation loss decreased to {val_loss:.6f}. Saving model...")
        self.best_model = model.state_dict()

    def load_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)


def collate_fn(batch):
    data, labels = zip(*batch)
    max_length = max(d.shape[0] for d in data)
    padded_data = [np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant') for d in data]
    return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def LOG_INFO(msg, mcolor='blue'):
    print(colored("#LOG :", 'green') + colored(msg, mcolor))


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
        choices=["AMDD"]
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument(
        "-sch", "--lr_scheduler", type=str,
        choices=["cos", "None",]
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0",
        help="Device to use, e.g. cuda:0, cuda:1, or cpu"
    )
    parser.add_argument(
        "-l", "--encoder_layers", type=int, default=8,
        help="Number of Transformer encoder layers"
    )
    parser.add_argument(
        "-ks", "--kernel_size", type=int, default=6,
        help="Temporal convolution kernel size"
    )
    parser.add_argument(
        "--loss_main_weight", type=float, default=0.9,
        help="Weight for the main fusion loss"
    )
    parser.add_argument(
        "--loss_aux_weight", type=float, default=0.05,
        help="Weight for each auxiliary branch loss (msa / vtt)"
    )
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Convert device string from CLI / config to torch.device
    if isinstance(args.device, list):
        args.device = args.device[0]
    args.device = torch.device(args.device)
    return args

def train_epoch(
    net, train_loader, loss_fn, optimizer, lr_scheduler, device, 
    current_epoch, total_epochs,
    loss_main_weight=0.9, loss_aux_weight=0.05,
):
    """One training epoch."""
    net.train()
    sample_count = 0
    running_loss = 0.
    running_msa_loss = 0.
    running_vtt_loss = 0.
    
    correct_count = 0
    msa_correct_count = 0
    vtt_correct_count = 0
    
    TP, FP, TN, FN = 0, 0, 0, 0
    msa_TP, msa_FP, msa_TN, msa_FN = 0, 0, 0, 0
    vtt_TP, vtt_FP, vtt_TN, vtt_FN = 0, 0, 0, 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch+1}/{total_epochs}",
        leave=False, unit="batch"
    ) as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            # pass to Model
            y_pred, msa_pred, vtt_pred = net(x)

            # Calculate losses
            loss = loss_fn(y_pred, y.to(torch.float32), net)
            loss_msa = loss_fn(msa_pred, y.to(torch.float32), net)
            loss_vtt = loss_fn(vtt_pred, y.to(torch.float32), net)
         #   torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            loss = (
                loss_main_weight * loss
                + loss_aux_weight * loss_msa
                + loss_aux_weight * loss_vtt
            )
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            # Update running losses
            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            running_msa_loss += loss_msa.item() * x.shape[0]
            running_vtt_loss += loss_vtt.item() * x.shape[0]

            # Calculate main model metrics
            pred = (y_pred > 0).int()
            correct_count += (pred == y).sum().item()

            TP += torch.sum((pred == 1) & (y == 1)).item()
            FP += torch.sum((pred == 1) & (y == 0)).item()
            TN += torch.sum((pred == 0) & (y == 0)).item()
            FN += torch.sum((pred == 0) & (y == 1)).item()

            # Calculate MSA model metrics
            msa_pred_bin = (msa_pred > 0).int()
            msa_correct_count += (msa_pred_bin == y).sum().item()

            msa_TP += torch.sum((msa_pred_bin == 1) & (y == 1)).item()
            msa_FP += torch.sum((msa_pred_bin == 1) & (y == 0)).item()
            msa_TN += torch.sum((msa_pred_bin == 0) & (y == 0)).item()
            msa_FN += torch.sum((msa_pred_bin == 0) & (y == 1)).item()

            # Calculate VTTEncoder model metrics
            vtt_pred_bin = (vtt_pred > 0).int()
            vtt_correct_count += (vtt_pred_bin == y).sum().item()

            vtt_TP += torch.sum((vtt_pred_bin == 1) & (y == 1)).item()
            vtt_FP += torch.sum((vtt_pred_bin == 1) & (y == 0)).item()
            vtt_TN += torch.sum((vtt_pred_bin == 0) & (y == 0)).item()
            vtt_FN += torch.sum((vtt_pred_bin == 0) & (y == 1)).item()

            # Calculate metrics
            def calculate_metrics(TP, FP, TN, FN):
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0
                return accuracy, precision, recall, f1_score

            main_acc, main_precision, main_recall, main_f1 = calculate_metrics(TP, FP, TN, FN)
            msa_acc, msa_precision, msa_recall, msa_f1 = calculate_metrics(msa_TP, msa_FP, msa_TN, msa_FN)
            vtt_acc, vtt_precision, vtt_recall, vtt_f1 = calculate_metrics(vtt_TP, vtt_FP, vtt_TN, vtt_FN)

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "loss_msa": running_msa_loss / sample_count,
                "loss_vtt": running_vtt_loss / sample_count,
                "acc": main_acc,
                "msa_acc": msa_acc,
                "vtt_acc": vtt_acc,
            })

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "loss": running_loss / sample_count,
        "loss_msa": running_msa_loss / sample_count,
        "loss_vtt": running_vtt_loss / sample_count,
        "acc": main_acc,
        "precision": main_precision, "recall": main_recall, "f1": main_f1,
        "msa_acc": msa_acc, 
        "msa_precision": msa_precision, "msa_recall": msa_recall, "msa_f1": msa_f1,
        "vtt_acc": vtt_acc, 
        "vtt_precision": vtt_precision, "vtt_recall": vtt_recall, "vtt_f1": vtt_f1,
    }

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

                ## amdd loss (total loss)
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

    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
    LOG_INFO(f"Using device: {args.device}")
    if torch.cuda.is_available():
        LOG_INFO(f"Available CUDA devices: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")

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
    for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        print(f"Fold {fold+1}: Train samples={len(train_indices)}, Val samples={len(val_indices)}")
        if len(val_indices) < args.batch_size:
            print(f"Warning: fold {fold+1} has insufficient validation samples.")

    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        LOG_INFO(f"Fold {fold+1}/{args.num_folds}")

        train_subset = Subset(combined_dataset, train_indices.tolist())
        val_subset = Subset(combined_dataset, val_indices.tolist())

        train_loader_fold = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,drop_last=True )
        val_loader_fold = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,drop_last=True)

        # Construct the model
        net = AMDD(
            d=256,
            l=args.encoder_layers,
            temporal_kernel_size=args.kernel_size,
        )

        net = net.to(args.device)
        LOG_INFO(f"[{args.model}] Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}", "cyan")
        LOG_INFO(f"Model device: {next(net.parameters()).device}")
        LOG_INFO(
            f"encoder_layers={args.encoder_layers}, kernel_size={args.kernel_size}, "
            f"loss_weights=(main={args.loss_main_weight}, aux={args.loss_aux_weight})",
            "cyan",
        )
        
        # Set other training components
        loss_fn = CombinedLoss(lambda_reg=1e-5, focal_weight=0.5, l2_weight=0.5)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,
                                 betas=(0.90, 0.9999),
                                 eps=1e-8,
                                 weight_decay=0.1,
                                 amsgrad=False
                                 )

        if args.lr_scheduler == "cos":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                #optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20
                optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20
            )
        else:
            lr_scheduler = None

        early_stopping = EarlyStopping(patience=5, verbose=True, save_path=os.path.join("./weights", f"best_model_wo_lrs_{fold}.pt"))

        # Training loop
        best_val_acc = -1.0
        for epoch in range(args.epochs):
            train_results = train_epoch(
                net, train_loader_fold, loss_fn, optimizer, lr_scheduler, 
                args.device, epoch, args.epochs,
                loss_main_weight=args.loss_main_weight,
                loss_aux_weight=args.loss_aux_weight,
            )
            val_results = val(net, val_loader_fold, loss_fn, args.device)

            print()
            LOG_INFO(f"Epoch: {epoch+1}/{args.epochs}", 'blue')
            LOG_INFO(f"Train Loss: {train_results['loss']:.4f}, Train Acc: {train_results['acc']:.4f}")
            LOG_INFO(f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['acc']:.4f}", mcolor="red")
            LOG_INFO(f"Val Precision: {val_results['precision']:.4f}, Val Recall: {val_results['recall']:.4f}, Val F1: {val_results['f1']:.4f}", mcolor="red")
            LOG_INFO(f"Train=> MSA Loss: {train_results['loss_msa']:.4f}, MSA Acc: {train_results['msa_acc']:.4f}, MSA precision: {train_results['msa_precision']:.4f}, MSA recall: {train_results['msa_recall']:.4f}, MSA F1: {train_results['msa_f1']:.4f}")
            LOG_INFO(f"Val=> MSA Loss: {val_results['loss_msa']:.4f}, MSA Acc: {val_results['msa_acc']:.4f}, MSA precision: {val_results['msa_precision']:.4f}, MSA recall: {val_results['msa_recall']:.4f}, MSA F1: {val_results['msa_f1']:.4f}", mcolor="red")
            LOG_INFO(f"Train=> VTTEncoder Loss: {train_results['loss_vtt']:.4f}, VTTEncoder Acc: {train_results['vtt_acc']:.4f}, VTTEncoder precision: {train_results['vtt_precision']:.4f}, VTTEncoder recall: {train_results['vtt_recall']:.4f}, VTTEncoder F1: {train_results['vtt_f1']:.4f}")
            LOG_INFO(f"Val=> VTTEncoder Loss: {val_results['loss_vtt']:.4f}, VTTEncoder Acc: {val_results['vtt_acc']:.4f}, VTTEncoder precision: {val_results['vtt_precision']:.4f}, VTTEncoder recall: {val_results['vtt_recall']:.4f}, VTTEncoder F1: {val_results['vtt_f1']:.4f}", mcolor="red")
            print()

            val_acc = val_results["acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save the best model
                torch.save(net.state_dict(), os.path.join("./weights", f"best_model_wo_lrs_{fold}.pt"))
                LOG_INFO(f"Best Model Saved in Epoch: {epoch+1}", mcolor='red')
            
            if early_stopping.early_stop:
                LOG_INFO("Early stopping triggered", 'red')
                break

        # Load the best model for testing
        net.load_state_dict(
            torch.load(os.path.join("./weights", f"best_model_wo_lrs_{fold}.pt"), map_location=args.device)
        )
        test_results = val(net, val_loader_fold, loss_fn, args.device)
        fold_results.append(test_results)
        LOG_INFO(f"Fold {fold+1} test results: {fold_results[-1]}", mcolor='yellow')

    print("============---------------================")
    LOG_INFO("Overall Folds Results for the test sets")
    LOG_INFO(f"All Results: {fold_results}")

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

    LOG_INFO(f"Average cross-validated results: {avg_results}", mcolor='green')
    result_file = "final_results.log"
    with open(result_file, 'w') as f:
        f.write("=== Final Evaluation Results ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in avg_results.items():
            f.write(f"{key}: {value}\n")
    
    LOG_INFO(f"Results saved to {result_file}", mcolor='green')

if __name__ == "__main__":
    main()