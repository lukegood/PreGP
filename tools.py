from itertools import islice
import torch
from torch.utils.data import DataLoader
from copy import deepcopy


def seq2kmer(seq, k):  # Perform k-mer processing
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    # kmers = " ".join(kmer)
    # return kmers

    # kmers = []
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    # for sub_list in kmer:
    #     sub_kmer = "".join(sub_list)
    #     kmers.append(sub_kmer)
    # return kmers
    kmers = str("")
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    for subList in kmer:
        subKmer = "".join(subList) + " "
        kmers += subKmer
    return kmers


def seq2kmer_small_seq(seqs, k):  # Perform k-mer processing
    kmers_list = []
    for seq in seqs:
        kmers = str("")
        kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
        for subList in kmer:
            subKmer = "".join(subList) + " "
            kmers += subKmer
        kmers_list.append(kmers)
    return kmers_list


def get_parameter_number(model):  #    Calculate parameter count
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_memory_usage(model, dtype_bytes=4):
    stats = get_parameter_number(model)
    total_mem = stats['Total'] * dtype_bytes  # Default float32 occupies 4 bytes
    return f"{total_mem / 1e6:.2f} MB"


def process_data_by_chr(ori_path, data_path):  # Process data by chromosome
    # ori_path = "./zc1404-207-delete.vcf"
    # data_path = "./gene-1404_42938.csv"
    # new_data_path = "./gene-1404_42938_new.csv"
    ori_path = ori_path
    data_path = data_path

    dna_dict = {}
    dnanum_dict = {}
    with open(ori_path, 'r') as file:  #    Generate sequence dictionary
        for line in islice(file, 1, None):
            list = line.split("	")
            num = list[1]
            name = list[3]
            dna_dict[name] = num
        for k, v in dna_dict.items():
            if v in dnanum_dict:
                dnanum_dict[v] += 1
            else:
                dnanum_dict[v] = 1

    new_snp_dict = {}
    with open(data_path, 'r') as f2:
        for line in islice(f2, 1, None):
            list_snp = line.split(",")
            name = list_snp[0]
            list_snp = list_snp[1:]
            start = 0
            end = 0
            new_snp_dict[name] = []
            for i in range(10):
                end = start + dnanum_dict[str(i + 1)] + 1
                sub_list = list_snp[start:end]
                new_snp_dict[name].append(sub_list)
                start = end

    return new_snp_dict


def check_whether_same_dynamic(output_index, targets, list_base, size):
    same_num = 0
    same_mask_num = 0
    sum = 0
    mask_sum = 0
    # mask_targets = []
    # mask_outputs_index = []
    for j in range(len(output_index)):
        base = list_base[j]
        mask_sub_outputs_index = output_index[j][base:base + size]
        mask_sub_targets = targets[j][base:base + size]
        # mask_outputs_index.append(mask_sub_outputs_index)
        # mask_targets.append(mask_sub_targets)
        sub_outputs_index = output_index[j]
        sub_tagets = targets[j]
        for i in range(len(mask_sub_outputs_index)):
            mask_sum += 1
            if mask_sub_outputs_index[i] == mask_sub_targets[i]:
                same_mask_num += 1
        for k in range(len(sub_outputs_index)):
            sum += 1
            if sub_outputs_index[k] == sub_tagets[k]:
                same_num += 1
    same_rate = same_num / sum
    mask_same_rate = same_mask_num / mask_sum
    # return same_rate, mask_same_rate, torch.stack(mask_targets), torch.stack(mask_outputs_index)
    return same_rate, mask_same_rate


# def check_whether_same(output_index, targets, list_base, size):
#     output_index = output_index.tolist()
#     targets = targets.tolist()
#     same_num = 0
#     same_mask_num = 0
#     sum = 0
#     mask_sum = 0
#     for j in range(len(output_index)):
#         base = list_base[j]
#         mask_list1 = output_index[j][base:base + size]
#         mask_list2 = targets[j][base:base + size]
#         sub_list1 = output_index[j]
#         sub_list2 = targets[j]
#         for i in range(len(mask_list1)):
#             mask_sum += 1
#             if mask_list1[i] == mask_list2[i]:
#                 same_mask_num += 1
#         for k in range(len(sub_list1)):
#             sum += 1
#             if sub_list1[k] == sub_list2[k]:
#                 same_num += 1
#     same_rate = same_num / sum
#     mask_same_rate = same_mask_num / mask_sum
#     return same_rate, mask_same_rate


def check_whether_same2(output_index, targets, base, size):
    mask_targets = targets[:, base:base + size]
    mask_outputs = output_index[:, base:base + size]
    same_num = 0
    sum = 0
    for i in range(len(mask_targets)):
        for j in range(len(mask_outputs[i])):
            sum += 1
            if mask_targets[i][j] == mask_outputs[i][j]:
                same_num += 1
    mask_same_rate = same_num / sum
    return mask_same_rate, mask_targets, mask_outputs


# def check_whether_same2(list1, list2, base, size):
#     list1 = list1.tolist()
#     list2 = list2.tolist()
#     same_num = 0
#     sum = 0
#     for j in range(len(list1)):
#         sub_list1 = list1[j][base:base + size]
#         sub_list2 = list2[j][base:base + size]
#         for i in range(len(sub_list1)):
#             sum += 1
#             if sub_list1[i] == sub_list2[i]:
#                 same_num += 1
#     same_rate = same_num / sum
#     return same_rate


def calculate_mask_seq(outputs, targets, base, size):
    # outputs = outputs.tolist()
    # targets = targets.tolist()
    mask_outputs = []
    mask_targets = []
    for i in range(len(base)):
        mask_targets.append(targets[i, base[i]:base[i] + size])
        mask_outputs.append(outputs[i, base[i]:base[i]+size, :])
        # mask_outputs.append(outputs[i][base[i]:base[i] + size])
    return torch.stack(mask_outputs), torch.stack(mask_targets)


def calculate_ignore_targets(targets, base, size):
    ignore_targets = targets.clone().detach()
    for seq in ignore_targets:
        for i in range(base):
            seq[i] = -100
        for i in range(base+size, len(seq)):
            seq[i] = -100
    return ignore_targets


def calculate_ignore_targets_small_seq(targets, base, size, small_seq, times, end_index):
    ignore_targets = targets.clone().detach()
    for seq in ignore_targets:
        if seq[base] == 8 or seq[base+1] == 11:  #    Quick check to determine if the base position is padding. If so, it means no mask was applied, only mark the padding part. This value needs to be modified according to the vocabulary
            for i in range((end_index-small_seq*(times-1))-2+1, len(seq)):  # -2 because kmer=3, +1 to skip cls ((end_index-2)-small_seq*(times-1))+1, len(seq)
                seq[0] = -100
                seq[i] = -100
        elif (base+size) < len(seq):
            for i in range(base):
                seq[i] = -100
            for i in range(base+size, len(seq)):
                seq[i] = -100
        elif (base+size) >= len(seq):
            for i in range(base):
                seq[i] = -100
    return ignore_targets


def calculate_ignore_targets_dynamic(targets, list_base, size):
    ignore_targets = targets.clone().detach()
    for seq, base in zip(ignore_targets, list_base):
        for i in range(base):
            seq[i] = -100
        for i in range(base+size, len(seq)):
            seq[i] = -100
    return ignore_targets


def mix_sentences(sentences):
    # sentences.to('cpu')
    base_item = deepcopy(sentences[0]).to('cpu')
    for i in range(1, len(sentences)):
        base_item['input_ids'] = torch.cat((base_item['input_ids'], sentences[i]['input_ids'].to('cpu')), dim=0)
        base_item['token_type_ids'] = torch.cat((base_item['token_type_ids'],sentences[i]['token_type_ids'].to('cpu')), dim=0)
        base_item['attention_mask'] = torch.cat((base_item['attention_mask'],sentences[i]['attention_mask'].to('cpu')), dim=0)
    return base_item
