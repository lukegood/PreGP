from transformers.models.bert import BertConfig, BertModel, BertTokenizer, BertForMaskedLM
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import numpy as np  # Import numpy library
import torch  # Import torch library
import torch.nn as nn  # Import torch.nn library
import tools
import argparse
import csv
from itertools import islice
import logging
from dataset import MyDatasetNewV3WithoutBigBatchPreTrainV2
from torch.utils.data import DataLoader
import torch.optim as optim  # Import optimizer
import pandas as pd
from collections import Counter
import copy
import math
import random
import time
import copy
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from memory import GPUMemoryReserver
import matplotlib.pyplot as plt

def cut_and_kmer(origin_data, cut_length, k, bag_num):
    # Split data
    cut_list = []
    for _, line in origin_data.items():
        data = line.strip().split(",")
        times = math.ceil(len(data) / cut_length)
        start = 0
        for _ in range(times):
            end = start + cut_length
            sub_list = data[start:end]
            # Reconnect with comma and write to file
            cut_list.append(",".join(sub_list))
            start = end
    # Perform k-mer conversion on the split data
    kmer_list = []
    for seq in cut_list:
        seq = seq.strip().replace(",", "")
        kmers = [seq[x : x + (k * bag_num)] 
             for x in range(0, len(seq) - k * bag_num + bag_num, bag_num)]
        kmer_list.append(" ".join(kmers))
    return kmer_list


MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}

def cal_perprexity_accuracy(logits, targets):
    # Calculate perplexity and mask accuracy
    probs = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, embedding_size]
    predicted_labels = torch.argmax(probs, dim=-1)

    # Extract token prediction probabilities
    size1 = targets.size(0)
    size2 = targets.size(1)
    prob_list = []
    ans_cor_list = []
    pred_list = []
    for i in range(size1):
        for j in range(size2):
            ans = targets[i][j].item()  # Correct answer
            if ans != -100:
                num = probs[i][j][ans]  # Prediction probability for the correct answer
                ans_pred = predicted_labels[i][j]  # Predicted value at this position
                prob_list.append(num)  # List of prediction probabilities for correct answers
                ans_cor_list.append(targets[i][j])  # List of correct answers
                pred_list.append(ans_pred)  # List of predicted answers

    # Ensure all tensors are on the same device and are float type
    prob_tensor = torch.stack(prob_list).float()
    ans_cor_tensor = torch.stack(ans_cor_list)
    ans_pred_tensor = torch.stack(pred_list)

    log_probs = torch.log(prob_tensor)
    total_loss = log_probs.sum()
    num_masked_tokens = len(ans_cor_list)
    
    # Use tensor calculation, avoid using .item()
    batch_perplexity = torch.exp(-total_loss / num_masked_tokens)

    # Calculate mask accuracy
    compare_result = ans_cor_tensor == ans_pred_tensor
    same_count = compare_result.sum()
    batch_accuracy = same_count.float() / len(ans_cor_list)

    return batch_perplexity, batch_accuracy

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


def create_logger(log_file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d %I:%M:%S %p')
    logger = get_logger(__name__)
    return logger


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Bert-Gene")

    parser.add_argument('--geno_path', type=str, default="./jianji_cut_2576_3mer.csv", help='path of geno file')
    parser.add_argument('--geno_path2', type=str, default="./jianji_cut_2576_3mer.csv", help='path of geno file')
    parser.add_argument('--chr_id_path', type=str, default="./chr_split_idx_1288_3mer.csv", help='path of geno file')
    parser.add_argument('--snp_id_path', type=str, default="./po_split_idx_1288_3mer.csv", help='path of geno file')
    parser.add_argument('--test_geno_path', type=str, default="./cut_2576_3mer.csv", help='path of geno file')
    parser.add_argument('--test_chr_id_path', type=str, default="./split_idx_1288_3mer.csv", help='path of geno file')
    parser.add_argument('--test_snp_id_path', type=str, default="./split_idx_1288_3mer.csv", help='path of geno file')
    parser.add_argument('--env_name', type=str, default='JL', help='name of phenotype')
    parser.add_argument('--pheno_name', type=str, default='PH', help='name of phenotype')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch', type=int, default=1, help='batchsize')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--depth', type=int, default=1, help='depth')
    parser.add_argument('--neurons1', type=int, default=512, help='neurons1 number')
    parser.add_argument('--neurons2', type=int, default=32, help='neurons2 number')
    parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
    parser.add_argument('--kmer_k', type=int, default=3)
    parser.add_argument('--d_embedding', type=int, default=768)
    parser.add_argument('--pretrain_model_path', type=str, default='./pretrain_model')
    parser.add_argument('--fine_tuning_model', type=str, default='./fine_tuning_model')
    parser.add_argument('--end_index', type=int, default=42938)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--mask_base', type=int, default=10)
    parser.add_argument('--special_word_size', type=int, default=5)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--range_num', type=int, default=10)
    parser.add_argument('--cut_length', type=int, default=1, help='batchsize')
    parser.add_argument('--log_file', type=str, default='info-newer.log')
    parser.add_argument('--run_log_path', type=str, default='/home/pod/shared-nvme/wanglg/running/run_log')
    parser.add_argument('--vocab_path', type=str, default='/home/pod/shared-nvme/wanglg/running/codes/all_standard')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--checkpoint_save_path', type=str, default='/home/pod/shared-nvme/wanglg/running/codes/all_standard')
    parser.add_argument('--save_interval', type=int, default=20000, help='save_interval')
    parser.add_argument('--checkpoint_load_file_path', type=str, default='/home/pod/shared-nvme/wanglg/running/codes/all_standard')
    parser.add_argument('--checkpoint_load_file_name', type=str, default='example.pth')
    parser.add_argument('--bag_num', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=1000, help='evaluation frequency (number of batches)')
    parser.add_argument('--reserved_memory', type=int, default=35000, help='evaluation frequency (number of batches)')
    args = parser.parse_args()
    return args


def make_collate_fn(vocab_size_base,cut_length, tokenizer, mask_ratio, kmer_k):
    # Note: cut_length refers to the sequence length calculated in tokens, excluding cls and sep
    cut_length = cut_length - kmer_k + 1
    def collate_fn(batch_samples):
        sentences = [sample for sample in batch_samples]
        encoded_text = tokenizer(
            sentences,
            padding="max_length",
            max_length=cut_length+2,
            return_tensors="pt"
        )

        mask_list = MASK_LIST[str(kmer_k)]

        if tokenizer.mask_token is None:
            raise ValueError("This tokenizer does not have a mask token.")

        labels = encoded_text['input_ids'].clone()
        probability_matrix = torch.full(labels.shape, mask_ratio)
        special_tokens_mask = torch.tensor(
            [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=torch.bool
        )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if tokenizer._pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Expand the scope of mask_list
        mask_offsets = torch.tensor(mask_list)
        for i in range(labels.size(0)):
            center_indices = torch.where(masked_indices[i])[0]
            if center_indices.numel() > 0:
                new_centers = (center_indices.unsqueeze(1) + mask_offsets).flatten()
                new_centers = new_centers[(new_centers >= 1) & (new_centers < labels.size(1) - 1)]
                masked_indices[i][torch.unique(new_centers)] = True

        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        encoded_text['input_ids'][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size_base, labels.shape, dtype=torch.long)
        encoded_text['input_ids'][indices_random] = random_words[indices_random]

        indices_unchanged = masked_indices & ~indices_replaced & ~indices_random

        return encoded_text, labels

    return collate_fn

# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, step, save_path, logger, TIMESTAMP):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step
    }
    checkpoint_file = f"{TIMESTAMP}_checkpoint_epoch_{epoch + 1}_step_{step}.pth"
    save_path = os.path.join(save_path, checkpoint_file)
    torch.save(checkpoint, save_path)
    logger.info(f"epoch {epoch + 1} step {step} Checkpoint saved at {save_path}")

# Load checkpoint
def load_checkpoint(model, optimizer, scheduler, load_path, load_name, logger):
    checkpoint_path = os.path.join(load_path, load_name)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    logger.info(f"Checkpoint loaded from {checkpoint_path}")  # Modified to use actual path
    return epoch, step

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
    set_all_seeds(42)
    args = create_arg_parser()
    logger = create_logger(args.log_file)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    current_file_name = os.path.basename(__file__)
    save_model_name = TIMESTAMP + "_" + str(args.env_name) + "_" + current_file_name
    
    # Ensure all required paths exist
    os.makedirs(args.run_log_path, exist_ok=True)
    os.makedirs(args.vocab_path, exist_ok=True)
    os.makedirs(args.pretrain_model_path, exist_ok=True)
    os.makedirs(args.checkpoint_save_path, exist_ok=True)
    
    logger.info(f"current_file_name:{current_file_name}")
    logger.info(f"save_model_name:{save_model_name}")
    logger.info(f"---------args in this run print start-------------")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"{arg_name}: {arg_value}")
    logger.info(f"---------args in this run print end-------------")
    logger.info("start load data.")

    #    Set tensorboard folder
    if accelerator.is_local_main_process:
        pic_log_path = os.path.join(args.run_log_path, "run_log" + "_" + TIMESTAMP + "_" + current_file_name)
        os.makedirs(pic_log_path, exist_ok=True)  # Create folder if it doesn't exist

    if not os.path.exists(args.pretrain_model_path):  # If path does not exist
        logger.info("path not found, created!!")
        os.makedirs(args.pretrain_model_path)

    # Create GPU memory reserver to avoid insufficient memory during preprocessing
    reserver = GPUMemoryReserver()
    reserver.reserve(args.reserved_memory)

    # Read genotype data
    geno_samples = []
    geno_data = {}
    for file_path in args.geno_path.split(','):
        file_path = file_path.strip()
        with open(file_path) as file:
            for line in islice(file, 1, None):
                parts = line.strip().split(",", 1)
                sample_id = parts[0]
                geno_samples.append(sample_id)
                geno_data[sample_id] = parts[1]
    
    # Split training and test sets
    test_ratio = 0.1  # Test set ratio
    all_samples = list(geno_data.keys())
    random.shuffle(all_samples)  # Random seed already set, should be reproducible
    split_idx = int(len(all_samples) * (1 - test_ratio))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    # Create training and test set data dictionaries
    train_data = {sample: geno_data[sample] for sample in train_samples}
    test_data = {sample: geno_data[sample] for sample in test_samples}
    train_snp_list_all = cut_and_kmer(train_data, args.cut_length, args.kmer_k, bag_num = args.bag_num)
    test_snp_list_all = cut_and_kmer(test_data, args.cut_length, args.kmer_k, bag_num = args.bag_num)
    logger.info("finish load geno data.")

    # snp_list_all = snp_list_all[:1000]


    special_word_size = args.special_word_size

    best_loss = 1.1e9

    # --------Split training and test sets---------------------------------
    val_data = test_snp_list_all
    train_data = train_snp_list_all
    # ---------------------------------------------------------------

    list_kmer_train = train_data
    list_kmer_val = val_data
    logger.info("kmer was done in dataset, skip kmer.")


    seq_len = max([len(sentence.split()) for sentence in list_kmer_train]) + 2  # +2 for cls and sep
    # seq_len = args.cut_length
    vocab = create_vocabulary(list_kmer_train)  # Create vocabulary for source and target languages, excluding special tokens like cls, sep
    vocab_size = len(vocab) + special_word_size
    # vocab_size = len(vocab)

    #    Generate vocabulary file
    vocab_file_name = TIMESTAMP + "_vocab.txt"
    if accelerator.is_local_main_process:
        # Ensure target directory exists
        os.makedirs(args.vocab_path, exist_ok=True)

        # Write new vocabulary
        vocab_file_path = os.path.join(args.vocab_path, vocab_file_name)
        with open(vocab_file_path, "w") as f2:
            for k, v in vocab.items():
                f2.write(k + "\n")

        # Append base vocabulary
        vocab_base_path = os.path.join(args.vocab_path, "vocab_base.txt")
        with open(vocab_base_path, "r") as f1:  # Open file
            data = f1.read()  # Read file
            with open(vocab_file_path, "a") as f2:  # Append mode
                f2.write(data)
    accelerator.wait_for_everyone()

    tokenizer = BertTokenizer(vocab_file=os.path.join(args.vocab_path, vocab_file_name), do_lower_case=False)
    logger.info("start create dataset.")
    dataset_train = MyDatasetNewV3WithoutBigBatchPreTrainV2(list_kmer_train, args.mask_ratio,
                                                            tokenizer, len(vocab), args.cut_length)
    dataset_test = MyDatasetNewV3WithoutBigBatchPreTrainV2(list_kmer_val, args.mask_ratio, tokenizer,
                                                           len(vocab), args.cut_length)
    logger.info("finish create dataset.")

    # Note: args.cut_length refers to sequence length calculated in tokens, excluding cls and seq
    collate_fn = make_collate_fn(len(vocab), args.cut_length, tokenizer, args.mask_ratio, args.kmer_k)

    train_loader = DataLoader(  #    Build torch DataLoader
        dataset=dataset_train,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(  #    Build torch DataLoader
        dataset=dataset_test,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn
    )

    hh = int(len(dataset_train))
    logger.info(f"Training set samples:{hh}")
    hh1 = int(len(dataset_test))
    print(f"Validation set samples:{hh1}")

    max_seq_len = seq_len  # Maximum sentence length (used for setting position encoding)
    logger.info(f" Corpus vocabulary size : {vocab_size}")  # Print vocabulary size
    logger.info(f" Maximum sentence length : {max_seq_len}")  # Print maximum sequence length

    # device = "cuda" if torch.cuda.is_available() else "cpu"  # Set device
    device = accelerator.device
    logger.info(f"device:{device}")

    reserver.release()

    bert_config = BertConfig(vocab_size=vocab_size,  # 32000
                             hidden_size=args.d_embedding,  # 4096//2
                             #  num_hidden_layers=12,  # 32//2
                             #  num_attention_heads=12,
                             max_position_embeddings=max_seq_len)  # 4096//2

    bert_model = BertForMaskedLM(config=bert_config).to(device)

    model = bert_model

    # para_num = tools.get_parameter_number(model)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Loss function
    # criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Optimizer

    # total_training_steps = args.epoch * len(train_loader) / 2 # Assume 100 epochs and train_loader length, actual step is half due to dual GPU
    total_training_steps = args.epoch * len(train_loader)
    num_warmup_steps = math.floor(total_training_steps * 0.06)  # Specify warmup steps
    logger.info(f"total_training_steps:{total_training_steps}")
    logger.info(f"num_warmup_steps:{num_warmup_steps}")
    # # Create warmup scheduler
    # scheduler_warmup = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    # Create linear decay scheduler
    scheduler_decay = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                      num_training_steps=total_training_steps)

    #---------------------Checkpoint loading related----------------------------------
    start_epoch, start_step = 0, 0  # Start from scratch
    if args.resume:
        # Load checkpoint (if exists)
        try:
            start_epoch, start_step = load_checkpoint(model, optimizer, scheduler_decay,
                                                    args.checkpoint_load_file_path,
                                                    args.checkpoint_load_file_name,
                                                    logger)
            logger.info(f"Resuming training from epoch {start_epoch + 1}, step {start_step}")
        except FileNotFoundError:
            logger.info("No checkpoint found. Starting from scratch.")
    #--------------------------------------------------------------------

    model, optimizer, train_loader, test_loader, scheduler_decay = accelerator.prepare(model,
                                                                                       optimizer,
                                                                                       train_loader,
                                                                                       test_loader,
                                                                                       scheduler_decay)
    total_training_steps = args.epoch * len(train_loader)
    num_warmup_steps = math.floor(total_training_steps * 0.06)  # Specify warmup steps
    logger.info(f"total_training_steps:{total_training_steps}")
    logger.info(f"num_warmup_steps:{num_warmup_steps}")

    start_time = time.time()
    epochs = args.epoch  # Training epochs
    global_step = start_step  # Start from resumed training step

    # Prepare data for saving loss curves
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch, epochs):  # Train for epochs rounds
        model.train()
        logger.info("Start training!")
        epoch_train_loss = 0.0

        for batch_idx, (inputs_data, targets_data) in enumerate(train_loader):
            with accelerator.accumulate(model):
                global_step += 1
                optimizer.zero_grad()
                inputs = inputs_data
                targets = targets_data.clone().detach()
                outputs = model(**inputs, labels=targets)
                loss = outputs.loss
                logits = outputs.logits
                epoch_train_loss += loss.item()  # For plotting loss curves

                if batch_idx % args.eval_freq == 0:
                    # Calculate mask accuracy and perplexity
                    batch_perplexity, batch_accuracy = cal_perprexity_accuracy(logits, targets)
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch: {epoch + 1:04d} batch:{batch_idx:04d} train cost = {loss:.6f} lr:{current_lr:.9f} batch_perplexity = {batch_perplexity:.4f} batch_accuracy = {batch_accuracy:.4f}")

                accelerator.backward(loss)
                optimizer.step()  # Update parameters
                scheduler_decay.step()

                if accelerator.is_local_main_process and global_step % args.save_interval == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_checkpoint(unwrapped_model, optimizer, scheduler_decay,
                                    epoch, global_step,
                                    args.checkpoint_save_path, logger, TIMESTAMP)

        # Calculate average training loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        loss_all = []
        perplexity_all = []
        accuracy_all = []
        logger.info("Start val!")
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (inputs_data, targets_data) in enumerate(test_loader):
                inputs = inputs_data
                targets = targets_data.clone().detach()
                outputs = model(**inputs, labels=targets)
                loss = outputs.loss
                logits = outputs.logits
                epoch_val_loss += loss.item()
                
                batch_perplexity, batch_accuracy = cal_perprexity_accuracy(logits, targets)
                loss_all.append(accelerator.gather_for_metrics(loss))
                perplexity_all.append(accelerator.gather_for_metrics(batch_perplexity))
                accuracy_all.append(accelerator.gather_for_metrics(batch_accuracy))

                if batch_idx % args.eval_freq == 0:
                    logger.info(f"Epoch: {epoch + 1:04d} step val batch:{batch_idx:04d} val cost = {loss:.6f} batch_perplexity = {batch_perplexity:.4f} batch_accuracy = {batch_accuracy:.4f}")

        losses = [loss.unsqueeze(0) if loss.dim() == 0 else loss for loss in loss_all]
        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        val_perplexityes = [val_perplexity.unsqueeze(0) if val_perplexity.dim() == 0 else val_perplexity for val_perplexity in perplexity_all]
        val_perplexityes = torch.cat(val_perplexityes)
        eval_val_perplexityes = torch.mean(val_perplexityes)

        accuracyes = [accuracy.unsqueeze(0) if accuracy.dim() == 0 else accuracy for accuracy in accuracy_all]
        accuracyes = torch.cat(accuracyes)
        eval_accuracy = torch.mean(accuracyes)

        avg_val_loss = epoch_val_loss / len(test_loader)
        test_losses.append(avg_val_loss)

        logger.info(f"Epoch: {epoch + 1:04d} epoch val cost = {eval_loss:.6f} epoch val perplexity = {eval_val_perplexityes:.6f} epoch val accuracy = {eval_accuracy:.6f}")

        # Save global best model
        if accelerator.is_local_main_process and eval_loss < best_loss:
            best_loss = eval_loss
            # best_model = copy.deepcopy(model)
            unwrap_model = accelerator.unwrap_model(model)
            torch.save(unwrap_model.state_dict(), os.path.join(args.pretrain_model_path, f"{save_model_name}_best_model.pth"))
            logger.info(f"best_loss:{best_loss}, best model saved!")
            logger.info(f"epoch:{epoch+1},saved best model")

        logger.info(f"epoch:{epoch+1} finished!")

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

        # Save image
        plt.tight_layout()
        plot_path = os.path.join(pic_log_path, f"loss.png")
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Loss and PCC curves saved to {plot_path}")

    # accelerator.wait_for_everyone()
    # if accelerator.is_local_main_process:
    #     unwrap_model = accelerator.unwrap_model(best_model)
    #     torch.save(unwrap_model.state_dict(), args.pretrain_model_path + "/" + save_model_name + "_best" + ".pth")
    #     logger.info(f"best_loss:{best_loss}, best model saved!")

    accelerator.wait_for_everyone()
    end_time = time.time()
    all_time = end_time - start_time

    logger.info(f"all_time:{all_time} s---{all_time / 60} mins---{all_time / 3600} hours---{all_time / 3600 / 24} days")
    logger.info("best_loss = {:.2f}".format(best_loss))
    logger.info("Training & val finished.")


if __name__ == '__main__':
    main()
