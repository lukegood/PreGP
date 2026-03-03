import torch
from torch.utils import data
from collections import Counter
import math
base = 50
import random


class MyDatasetNewV3WithoutBigBatchPreTrainV2FineTuning(data.Dataset):
    def __init__(self, sentences, phe):
        self.sentences = sentences
        self.phe = phe

    def __getitem__(self, index):
        # return self.id[index], self.phe[index], self.dictseq[index]
        #return self.input_data[index], self.output_data[index], self.atten_mask_data[index], self.phe[index], self.id[index]
        return self.sentences[index], self.phe[index]

    def __len__(self):
        return len(self.sentences)
    
class MyDatasetNewV3SoftPrompt(data.Dataset):
    def __init__(self, sentences, phe, group_ids):
        self.sentences = sentences
        self.phe = phe
        self.group_ids = group_ids

    def __getitem__(self, index):
        # return self.id[index], self.phe[index], self.dictseq[index]
        #return self.input_data[index], self.output_data[index], self.atten_mask_data[index], self.phe[index], self.id[index]
        return self.sentences[index], self.phe[index], self.group_ids[index]

    def __len__(self):
        return len(self.sentences)

class MyDatasetNewV3WithoutBigBatchPreTrainV2FineTuningPosition(data.Dataset):
    def __init__(self, sentences, phe, chr_ids, snp_ids):
        self.sentences = sentences
        self.phe = phe
        self.chr_ids = chr_ids
        self.snp_ids = snp_ids

    def __getitem__(self, index):
        # return self.id[index], self.phe[index], self.dictseq[index]
        #return self.input_data[index], self.output_data[index], self.atten_mask_data[index], self.phe[index], self.id[index]
        return self.sentences[index], self.phe[index], self.chr_ids[index], self.snp_ids[index]

    def __len__(self):
        return len(self.sentences)
    
class MyDatasetNewV3WithoutBigBatchPreTrainV2FineTuningMultiGroup(data.Dataset):
    def __init__(self, sentences, phe, chr_ids, snp_ids, group_ids):
        self.sentences = sentences
        self.phe = phe
        self.chr_ids = chr_ids
        self.snp_ids = snp_ids
        self.group_ids = group_ids

    def __getitem__(self, index):
        # return self.id[index], self.phe[index], self.dictseq[index]
        #return self.input_data[index], self.output_data[index], self.atten_mask_data[index], self.phe[index], self.id[index]
        return self.sentences[index], self.phe[index], self.chr_ids[index], self.snp_ids[index], self.group_ids[index]

    def __len__(self):
        return len(self.sentences)
