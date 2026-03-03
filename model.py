import torch.nn as nn  # Import torch.nn library
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from transformers.models.bert import BertPreTrainedModel, BertModel

    
class FeatureGlobalv1NoChrSNP(nn.Module):
    def __init__(self, model, d_embedding, cut_length, small_batch,
                 fc2_hidden_dim=512,
                 fc3_hidden_dim=1024,
                 dropout_rate=0.5):
        super(FeatureGlobalv1NoChrSNP, self).__init__()
        self.bert = model
        #    Number of slices already calculated based on k-mer length
        self.small_batch = small_batch
        #    cut_length here is already the length after k-mer processing
        self.cut_length = cut_length
        self.d_embedding = d_embedding
        # self.projection_new = nn.Linear((max_seq_len - 1) * d_embedding, out_features=1)
        self.fc2 = nn.Sequential(
            nn.Linear(self.small_batch * cut_length, out_features=fc2_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2_hidden_dim, 1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(d_embedding, out_features=fc3_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc3_hidden_dim, fc3_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc3_hidden_dim, 1)
        )

    def forward(self, x):
        outputs = self.bert(**x)
        token_vector = outputs['last_hidden_state']
        token_vector_batch = self.process_data2(token_vector)
        
        token_vector_batch = token_vector_batch.transpose(1, 2)
        projection_p1 = self.fc2(token_vector_batch)  # [batch, 768, 1]
        projection_p1 = projection_p1.squeeze(-1)

        projection = self.fc3(projection_p1)
        return projection

    def process_data2(self, data):
        data = data.view(-1, self.small_batch * self.cut_length, self.d_embedding)
        return data
