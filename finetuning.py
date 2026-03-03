#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Liguang Wang
@license: (C) Since 2024, All rights reserved.
@Time: 2024/8/31 10:47
"""
from setproctitle import setproctitle
setproctitle("python")

import time
from transformers.models.bert import BertConfig, BertModel, BertTokenizer, BertForMaskedLM
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import numpy as np  # Import numpy library
import torch
import torch.nn as nn  # Import torch.nn library
import argparse
import csv
from itertools import islice
import logging
from dataset import MyDatasetNewV3WithoutBigBatchPreTrainV2FineTuning
from torch.utils.data import DataLoader
import torch.optim as optim  # Import optimizer
import pandas as pd
from collections import Counter
import copy
import math
from model import FeatureBase, FeatureGlobalv1, FeatureGlobalv1NoChrSNP, FeatureGlobalv1NoChrSNPMeanPool
from scipy.stats import pearsonr
import random
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from tools import get_parameter_number
import matplotlib.pyplot as plt
from memory import GPUMemoryReserver

# GPU memory cleanup
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


def create_vocabulary(sentences):
    vocab = {}
    counter = Counter()
    # Count word frequencies in the corpus
    for sentence in sentences:
        words = sentence.split()
        counter.update(words)
        # Create vocabulary and assign a unique index to each word
    for word in counter:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

@torch.no_grad()
def preencode_all(sentences, tokenizer, end_index, max_length, kmer_k):
    """Split and encode all samples at once, returning a list of encoded dictionaries organized by sample.
    - sentences: List[str] (each element is a complete k-mer string of a sample, space-separated)
    - tokenizer: HF tokenizer (BertTokenizerFast preferred)
    - end_index, max_length, kmer_k: Keep consistent with your original logic for calculating split times and sequence length
    Returns: List[Dict[str, Tensor]], each sample contains three tensors of shape [times, seq_len].
    """
    #   Calculate length after k-mer processing
    cut_length = max_length - kmer_k + 1
    #   Calculate number of slices based on k-mer length
    times = int(math.ceil((end_index - kmer_k + 1) / cut_length))
    #   Calculate segment length after adding special tokens at both ends
    seq_len = cut_length + 2 # + [CLS],[SEP]


    # Construct all subsequences (by sample -> times segments), batch tokenize at once
    list_all = []
    for sentence in sentences:
        new_sentence = sentence.split(" ")
        start_idx = 0
        for i in range(times):
            end_idx = start_idx + cut_length  #   Python will automatically truncate if the last segment is not long enough, no extra handling needed
            sub_list = new_sentence[start_idx:end_idx]
            sub_str = " ".join(sub_list)
            list_all.append(sub_str)
            start_idx = end_idx

    encoded = tokenizer(
        list_all,
        padding="max_length",
        max_length=cut_length+2,  #   extra cls and sep
        return_tensors="pt"
    )

    N = len(sentences)
    out = []
    for i in range(N):
        s = i * times
        e = s + times
        item = {
            "input_ids": encoded["input_ids"][s:e].contiguous(),
            "attention_mask": encoded["attention_mask"][s:e].contiguous(),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"]))[s:e].contiguous(),
        }
        out.append(item)
    return out, times, seq_len


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_per_sample, phe_list):
        self.encoded = encoded_per_sample
        self.phe = phe_list
    def __len__(self):
        return len(self.phe)
    def __getitem__(self, idx):
        return self.encoded[idx], float(self.phe[idx])


def create_logger(log_file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d %I:%M:%S %p')
    logger = get_logger(__name__)
    return logger


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Bert-Gene")

    parser.add_argument('--geno_path', type=str, default='D:\MyFiles\Research\Resource\Data\data6210\data6210-1_ID_kmer_all_42938.csv', help='path of geno file')
    parser.add_argument('--cvf_path', type=str, default='D:\MyFiles\Research\Resource\Data\data6210\CVF_new_MG_1541.csv', help='path of cvf file')
    parser.add_argument('--phe_path', type=str, default='D:/MyFiles/Research/Resource/Data/data6210/PH_phe.csv', help='path of phe file')
    parser.add_argument('--env_name', type=str, default='Jilin_PH', help='name of phenotype')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--big_batch', type=int, default=1, help='batchsize')
    parser.add_argument('--cut_length', type=int, default=1, help='batchsize')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--lr', type=float, default=1e-7, help='learning rate')
    parser.add_argument('--kmer_k', type=int, default=3)
    parser.add_argument('--d_embedding', type=int, default=768)
    parser.add_argument('--pretrain_model_path', type=str, default='./pretrain_model')
    parser.add_argument('--fine_tuning_model_path', type=str, default='./fine_tuning_model')
    parser.add_argument('--end_index', type=int, default=42938)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--special_word_size', type=int, default=5)
    parser.add_argument('--log_file', type=str, default='info-newer.log')
    parser.add_argument('--load_model_name', type=str, default='bestmodel_0827.pth')
    parser.add_argument('--run_log_path', type=str, default='/home/pod/shared-nvme/wanglg/running/run_log')
    parser.add_argument('--vocab_path', type=str, default='/home/pod/shared-nvme/wanglg/running/codes/all_standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--fold', type=int, default=1, help='fold')
    parser.add_argument('--premodel_vocab_size', type=int, default=4101, help='premodel_vocab_size')
    parser.add_argument('--vocab_name', type=str, default='2025-04-08T14-47-21_vocab.txt')
    parser.add_argument('--group_name', type=str, default='1404')
    parser.add_argument('--eval_freq', type=int, default=2000, help='evaluation frequency (number of batches)')
    parser.add_argument('--pred_save_path', type=str, default='./predictions', help='path to save prediction results')
    parser.add_argument('--n_folds', type=int, default=10, help='n folds')
    parser.add_argument('--bag_num', type=int, default=2, help='bag_num')
    parser.add_argument('--snp_chr_bag_num', type=int, default=1, help='snp_chr_bag_num')
    parser.add_argument('--reserved_memory', type=int, default=17000, help='reserved_memory')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='evaluation frequency (number of batches)')
    parser.add_argument('--trainable_layers', type=str, default='projection_new,fc,fc2,fc3', 
                        help='Comma-separated list of layer names to be trainable')
    parser.add_argument('--unfreeze_from_layer', type=int, default=11,
                        help='Unfreeze BERT layers starting from this index (0-based)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='early_stopping_patience')
    parser.add_argument('--random_seed', type=int, default=1234, help='fc2 hidden dimension')
    parser.add_argument('--fc2_hidden_dim', type=int, default=512, help='fc2 hidden dimension')
    parser.add_argument('--fc3_hidden_dim', type=int, default=1024, help='fc3 hidden dimension')
    parser.add_argument('--predict_drop_out_rate', type=float, default=0.5, help='prediction dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for Adam optimizer')
    parser.add_argument('--scheduler_type', type=str, default='linear', choices=['linear', 'cosine'], help='learning rate scheduler type')
    args = parser.parse_args()

    return args

def start_kmer(sentence, k, bag_num):
    for key, line in sentence.items():
        seq = line.replace("\n", "").replace(",", "")
        kmer = [seq[x:x + (k * bag_num)] for x in range(0, len(seq) - k * bag_num + bag_num, bag_num)]
        kmers = " ".join(kmer)
        sentence[key] = kmers
    return sentence

def collate_fn(batch):
    enc_list, phe_list = zip(*batch)
    ids = torch.vstack([x["input_ids"] for x in enc_list]) # [B*times, L]
    attn = torch.vstack([x["attention_mask"] for x in enc_list])
    tti = torch.vstack([x["token_type_ids"] for x in enc_list])
    phe = torch.tensor(phe_list, dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": attn, "token_type_ids": tti}, phe


def set_all_seeds(seed_value=42):
    """Set all random seeds to ensure experiment reproducibility"""
    set_seed(seed_value)  # Set accelerate random seed
    random.seed(seed_value)  # Python built-in random
    np.random.seed(seed_value)  # Numpy random
    torch.manual_seed(seed_value)  # PyTorch CPU random
    torch.cuda.manual_seed_all(seed_value)  # PyTorch GPU random
    # torch.backends.cudnn.deterministic = True  # Ensure CUDA convolution operations are reproducible
    # torch.backends.cudnn.benchmark = False  # Disable benchmark optimization for deterministic results


def main():
    args = create_arg_parser()
    set_all_seeds(args.random_seed)  # Call at the beginning of main function
    logger = create_logger(args.log_file)
    # kwargs = DDPK(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[kwargs])
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    current_file_name = os.path.basename(__file__)
    run_tag = f"{TIMESTAMP}_{args.env_name}"
    logger.info(f"run_tag:{run_tag}")
    
    logger.info(f"env_name:{args.env_name}")
    logger.info(f"pretrain model:{args.load_model_name}")
    logger.info(f"current_file_name:{current_file_name}")
    logger.info(f"---------args in this run print start-------------")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"{arg_name}: {arg_value}")
    logger.info(f"---------args in this run print end-------------")
    logger.info("start load data.")

    # Ensure all paths exist
    os.makedirs(args.run_log_path, exist_ok=True)
    os.makedirs(args.vocab_path, exist_ok=True)
    os.makedirs(args.pretrain_model_path, exist_ok=True)
    os.makedirs(args.fine_tuning_model_path, exist_ok=True)
    os.makedirs(args.pred_save_path, exist_ok=True)
    reserver = GPUMemoryReserver()
    reserver.reserve(args.reserved_memory)
    #   Set tensorboard folder
    if accelerator.is_local_main_process:
        pic_log_path = os.path.join(args.run_log_path, run_tag)  # Use os.path.join
        os.makedirs(pic_log_path, exist_ok=True)  # Create folder if it doesn't exist

    # Read vcf file
    df_cvf_list = []
    for file_path in args.cvf_path.split(','):
        file_path = file_path.strip()
        df_cvf_list.append(pd.read_csv(file_path))
    df_cvf = pd.concat(df_cvf_list, ignore_index=True)
    cvf_samples = df_cvf.iloc[:, 0].tolist()  # Get sample IDs from CVF file
    # Set of materials present in CVF for later filtering
    cvf_set = set(cvf_samples)

    # Read genotype data and establish sample ID to genotype mapping
    geno_samples = []
    geno_data = {}
    for file_path in args.geno_path.split(','):
        file_path = file_path.strip()
        with open(file_path) as file:
            for line in islice(file, 1, None):
                parts = line.strip().split(",", 1)
                sample_id = parts[0]
                if sample_id in cvf_set:
                    geno_samples.append(sample_id)
                    geno_data[sample_id] = parts[1]

    # Read phenotype data and establish sample ID to phenotype mapping
    phe_samples = []
    phe_data = {}
    env_names = [name.strip() for name in args.env_name.split(',')]
    phe_paths = [path.strip() for path in args.phe_path.split(',')]
    
    # If only one env_name is specified, use the same environment name for all files
    if len(env_names) == 1:
        env_names = env_names * len(phe_paths)
    elif len(env_names) != len(phe_paths):
        raise ValueError("Number of env_name must be 1 or match the number of phe_path")
    
    for file_path, env in zip(phe_paths, env_names):
        with open(file_path) as file:
            # Read header to get column names
            header = next(file).strip().split(',')
            try:
                col_idx = header.index(env)
            except ValueError:
                raise ValueError(f"Environment name {env} not found in phenotype file {file_path}")
                
            for line in file:
                parts = line.strip().split(",")
                sample_id = parts[0]
                # Only keep samples that exist in CVF
                if sample_id in cvf_set:
                        phe_samples.append(sample_id)
                        phe_data[sample_id] = float(parts[col_idx])
    logger.info("finish load phe data.")

    logger.info("Start converting data to k-mer form.")
    geno_data = start_kmer(geno_data, args.kmer_k, bag_num = args.bag_num)
    logger.info("Genotype data has been converted to k-mer form.")

    # Check if samples in CVF file can be found in genotype and phenotype files
    # Not all samples in genotype and phenotype files need to be in CVF file
    cvf_set = set(cvf_samples)
    geno_set = set(geno_samples)
    phe_set = set(phe_samples)

    # Check if samples in CVF file are in genotype file
    missing_in_geno = cvf_set - geno_set
    if missing_in_geno:
        logger.error(f"{len(missing_in_geno)} samples in CVF file not found in genotype file: {list(missing_in_geno)[:10]}...")
        raise ValueError("Samples in CVF file missing in genotype file")

    # Check if samples in CVF file are in phenotype file
    missing_in_phe = cvf_set - phe_set
    if missing_in_phe:
        logger.error(f"{len(missing_in_phe)} samples in CVF file not found in phenotype file: {list(missing_in_phe)[:10]}...")
        raise ValueError("Samples in CVF file missing in phenotype file")

    # Reorganize data according to CVF file order
    list_phe = [phe_data[sample] for sample in cvf_samples]
    snp_list_all = [geno_data[sample] for sample in cvf_samples]

    # Build tokenizer / dataset / dataloader
    tokenizer = BertTokenizer(
        vocab_file=os.path.join(args.vocab_path, args.vocab_name),
        do_lower_case=False,
    )
    logger.info("Start pre-encoding all sentences with tokenizer (one-time).")
    encoded_bank, times, seq_len = preencode_all(
        snp_list_all, tokenizer, args.end_index, args.cut_length, args.kmer_k
    )
    logger.info(f"Pre-encoding done: total samples={len(encoded_bank)}, times={times}, seq_len={seq_len}")

    fold_results = []
    for fold in range(1, args.n_folds + 1):
        # ==== 10-Fold CV BEGIN ====
        logger.info(f"========== Fold {fold}/{args.n_folds} Start ==========")
        test_idx = df_cvf[df_cvf["cv_1"] == fold].index
        train_idx = df_cvf[df_cvf["cv_1"] != fold].index
        test_sample_ids = [cvf_samples[i] for i in test_idx]
        # ==== 10-Fold CV END ====
        dataset_train = EncodedDataset(
            [encoded_bank[i] for i in train_idx],
            [list_phe[i] for i in train_idx]
        )
        dataset_test = EncodedDataset(
            [encoded_bank[i] for i in test_idx],
            [list_phe[i] for i in test_idx]
        )
        # num_workers = max(1, min(os.cpu_count() or 4, 8))
        #   Temporarily set to 2 to prevent speed issues
        num_workers = 2
        train_loader = DataLoader(
            dataset_train,
            batch_size=args.big_batch,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=args.big_batch,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
        )
        reserver.release()
        # Build model
        device = accelerator.device
        logger.info(f"device:{device}")
        bert_config = BertConfig(
            vocab_size=args.premodel_vocab_size,
            hidden_size=args.d_embedding,
            num_hidden_layers=args.num_hidden_layers,
            max_position_embeddings=args.cut_length - args.kmer_k + 1 + 2,
        )
        bert_model = BertModel(config=bert_config).to(accelerator.device)
        model = FeatureGlobalv1NoChrSNP(
            bert_model,
            args.d_embedding,
            args.cut_length - args.kmer_k + 1 + 2,
            int(math.ceil((args.end_index - args.kmer_k + 1) / (args.cut_length - args.kmer_k + 1))),
            fc2_hidden_dim = args.fc2_hidden_dim,
            fc3_hidden_dim = args.fc3_hidden_dim,
            dropout_rate = args.predict_drop_out_rate
        ).to(device)

        # Load pre-trained weights
        tmp_checkpoint = torch.load(os.path.join(args.pretrain_model_path, args.load_model_name))
        pretext_dict = tmp_checkpoint["model_state_dict"]
        model_dict = model.state_dict()
        for k, v in pretext_dict.items():
            if k in model_dict:
                model_dict[k] = v
                logger.info(f"Successfully matched parameter: {k}")
                if fold == 1:
                    logger.info(f"Successfully matched parameter: {k}, value:{v}")
            else:
                logger.warning(f"Key {k} from pre-trained model not found in current model, parameter not loaded.")
        model.load_state_dict(model_dict)
    
        # Freeze / Unfreeze
        trainable_layers = args.trainable_layers.split(',')
        for n, p in model.named_parameters():
            if any(x in n for x in trainable_layers):
                p.requires_grad = True
                logger.info(f"name:{n} is required grad. required grad status:{p.requires_grad}")
            else:
                p.requires_grad = False
                logger.info(f"name:{n} is not required grad. required grad status:{p.requires_grad}")
        for i, layer in enumerate(model.bert.encoder.layer):
            if i >= args.unfreeze_from_layer:
                logger.info(f"layer:{i} is unfrozen.")
                for p in layer.parameters():
                    p.requires_grad = True
                    logger.info(f"name:{p} is required grad. required grad status:{p.requires_grad}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = args.epoch * len(train_loader)
        warmup_steps = math.floor(total_steps * 0.06)
        if args.scheduler_type == 'linear':
            scheduler_decay = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif args.scheduler_type == 'cosine':
            scheduler_decay = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

        model, optimizer, train_loader, test_loader, scheduler_decay = accelerator.prepare(
            model, optimizer, train_loader, test_loader, scheduler_decay
        )

        # TensorBoard
        if accelerator.is_local_main_process:
            writer = SummaryWriter(os.path.join(args.run_log_path, f"run_log_{run_tag}_fold{fold}"))

        # Training
        best_pcc = -np.inf

        # Prepare data for saving loss curves
        train_losses = []
        test_losses = []
        train_pccs = []
        test_pccs = []

        patience_counter = 0
        for epoch in range(args.epoch):
            model.train()
            epoch_train_pccs = []
            epoch_train_loss = 0.0

            for batch_idx, (inputs_data, phe) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    outputs = model(inputs_data)
                    loss = criterion(outputs.view(-1), phe)
                    epoch_train_loss += loss.item()  # For plotting loss curves

                    if batch_idx % args.eval_freq == 0:

                        pred_all = accelerator.gather_for_metrics(outputs.view(-1))
                        pred_all = pred_all.cpu().detach().numpy().tolist()
                        phe_all = accelerator.gather_for_metrics(phe)
                        phe_all = phe_all.cpu().detach().numpy().tolist()
                        pred_train = np.asarray(pred_all)
                        phe_train = np.asarray(phe_all)

                        if len(pred_all) > 1 and len(phe_all) > 1:
                            pccs = pearsonr(pred_train.reshape(-1), phe_train.reshape(-1))
                            # Add PCC for plotting
                            epoch_train_pccs.append(pccs[0])
                        else:
                            logger.info("Input arrays must have at least 2 elements. skip!")
                            pccs = None

                        if pccs is not None:
                            logger.info(
                                f"[Fold {fold}] epoch {epoch+1} batch {batch_idx} train PCC={pccs[0]:.4f} pval={pccs[1]:.4f}"
                            )
                        else:
                            logger.info(
                                f"[Fold {fold}] epoch {epoch+1} batch {batch_idx} train PCC=N/A pval=N/A"
                            )

                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(f"Epoch: {epoch + 1:04d} batch:{batch_idx:04d} train cost = {loss:.6f} lr:{current_lr:.9f}")

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler_decay.step()
                    optimizer.zero_grad()

            
            # Calculate average training loss
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            avg_train_pcc = sum(epoch_train_pccs) / len(epoch_train_pccs) if epoch_train_pccs else 0
            train_pccs.append(avg_train_pcc)
            
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            all_val_pred, all_val_phe = [], []
            logger.info("Start val!")
            with torch.no_grad():
                for batch_idx, (inputs_data, phe) in enumerate(test_loader):
                    pred = model(inputs_data).view(-1)
                    loss = criterion(pred, phe)
                    epoch_val_loss += loss.item()

                    #   Distributed evaluation needs gather_for_metrics to aggregate data from multiple threads and remove duplicates
                    pred_all = accelerator.gather_for_metrics(pred)
                    pred_all = pred_all.cpu().detach().numpy().tolist()
                    all_val_pred.extend(pred_all)

                    phe_all = accelerator.gather_for_metrics(phe)
                    phe_all = phe_all.cpu().detach().numpy().tolist()
                    all_val_phe.extend(phe_all)

                    # if batch_idx % args.eval_freq == 0:
                    #     pred_all = np.asarray(pred_all)
                    #     phe_all = np.asarray(phe_all)
                    #     if len(pred_all) > 1 and len(phe_all) > 1:
                    #         pccs = pearsonr(pred_all.reshape(-1), phe_all.reshape(-1))
                    #     else:
                    #         logger.info("Input arrays must have at least 2 elements. skip!")
                    #         pccs = None

                    #     logger.info(f"Epoch: {epoch + 1:04d} batch:{batch_idx:04d} test batch pccs:{pccs}")
                    
                    #     logger.info(f"Epoch: {epoch + 1:04d} val batch:{batch_idx:04d} val cost = {loss:.6f}")


            # Calculate average validation loss
            avg_val_loss = epoch_val_loss / len(test_loader)
            test_losses.append(avg_val_loss)

            all_val_pred = np.asarray(all_val_pred)
            all_val_phe = np.asarray(all_val_phe)
            logger.info(f"all_val_pred:{all_val_pred}")
            logger.info(f"all_val_phe:{all_val_phe}")
            pccs = pearsonr(all_val_pred.reshape(-1), all_val_phe.reshape(-1))
            logger.info(f"[Fold {fold}] epoch {epoch+1} VAL PCC={pccs[0]:.4f}  p={pccs[1]:.4g}")
            # Add PCC for plotting
            test_pccs.append(pccs[0])

            if pccs[0] > best_pcc:
                patience_counter = 0
                best_pcc, best_p = pccs
                if accelerator.is_local_main_process:
                    unwrap_model = accelerator.unwrap_model(model)
                    torch.save(
                        unwrap_model.state_dict(),
                        os.path.join(args.fine_tuning_model_path, f"{run_tag}_fold{fold}_best.pth"),
                    )
                    pd.DataFrame(
                        {
                            "sample_id": test_sample_ids,
                            "pred_value": all_val_pred,
                            "true_value": all_val_phe,
                        }
                    ).to_csv(
                        os.path.join(args.pred_save_path, f"{run_tag}_fold{fold}_pred.csv"), index=False
                    )

                    logger.info(f"best_pcc:{best_pcc},p_value_0:{best_p}, fold {fold} best model saved!")
                    logger.info(f"best model saved to {args.fine_tuning_model_path}/{run_tag}_fold{fold}_best.pth")
                    logger.info(f"Predictions saved to {args.pred_save_path}/{run_tag}_fold{fold}_pred.csv")
                    logger.info(f"fold {fold}, epoch:{epoch+1}, saved best model")
            else:
                patience_counter += 1
            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        fold_results.append(best_pcc)

        if accelerator.is_local_main_process:

            plt.figure(figsize=(12, 6))
            
            # Plot loss curves
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Plot PCC curves
            plt.subplot(1, 2, 2)
            plt.plot(train_pccs, label='Train PCC')
            plt.plot(test_pccs, label='Test PCC')
            plt.xlabel('Epoch')
            plt.ylabel('PCC')
            plt.title('Training and Validation PCC')
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(pic_log_path, f"loss_pcc_curve_fold_{fold}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Loss and PCC curves saved to {plot_path}")
        logger.info(f"========== Fold {fold} Done — Final Best PCC={best_pcc:.4f} ==========")
        # Clean up GPU memory
        torch.cuda.empty_cache()



    # CV Summary
    if accelerator.is_local_main_process:
        fold_results = np.asarray(fold_results)
        logger.info("======== 10-Fold CV Summary ========")
        logger.info("PCCs: " + "  ".join(f"{x:.4f}" for x in fold_results))
        logger.info(f"Mean PCC = {fold_results.mean():.4f} ± {fold_results.std(ddof=1):.4f}")

if __name__ == "__main__":
    main()
