import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

# Set random seeds for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

relu = torch.nn.ReLU()
# Create Tanh layer
tanh_layer = nn.Tanh()

# New attention mechanism for testing
class CustomAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(CustomAttention, self).__init__()
        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, embeddings):
        queries = self.query_layer(embeddings)
        keys = self.key_layer(embeddings)
        values = self.value_layer(embeddings)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        return attention_output, attention_weights


class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, conv_kernel_sizes=[2, 4]):
        super(AttentionModel, self).__init__()

        # Create a convolutional layer for each kernel size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0)) for k in conv_kernel_sizes]
        )

        # Self-attention layer for intra-group context aggregation
        self.intra_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=len(conv_kernel_sizes))

        # Self-attention layer for inter-group context aggregation
        self.inter_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=len(conv_kernel_sizes))

        # Linear layer for final output
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, normalized_embeddings, group_labels):
        batch_size, embedding_dim = normalized_embeddings.size()
        combined_output = torch.zeros_like(normalized_embeddings)

        unique_groups = torch.unique(group_labels)

        for group in unique_groups:
            # Get current group's student indices and embeddings
            group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
            group_embeddings = normalized_embeddings[group_indices]  # Shape: (group_size, embedding_dim)
            group_size = group_embeddings.size(0)

            # Reshape group embeddings to (1, 1, group_size, embedding_dim) for 2D convolution
            group_embeddings = group_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, group_size, embedding_dim)

            # Generate summary tokens (intra-group context aggregation) through each conv layer
            aggregated_tokens = []
            for conv in self.convs:
                aggregated_token = conv(group_embeddings)  # Output shape: (1, 1, new_group_size, embedding_dim)
                aggregated_token = aggregated_token.squeeze(0).squeeze(0)  # Shape: (new_group_size, embedding_dim)

                # Pad or crop to match original group size
                if aggregated_token.size(0) < group_size:
                    # If conv output is shorter, pad with zeros at the end
                    pad_size = group_size - aggregated_token.size(0)
                    aggregated_token = nn.functional.pad(aggregated_token, (0, 0, 0, pad_size))
                else:
                    # If conv output is longer, crop to original size
                    aggregated_token = aggregated_token[:group_size]

                aggregated_tokens.append(aggregated_token)

            # Stack and concatenate outputs from different conv layers
            aggregated_tokens = torch.stack(aggregated_tokens, dim=1)  # (group_size, num_heads, embedding_dim)

            # Calculate intra-group self-attention with conv-processed representations
            attention_input = aggregated_tokens.transpose(0, 1)  # (num_heads, group_size, embedding_dim)
            intra_attention_output, _ = self.intra_attention_layer(
                query=attention_input,
                key=attention_input,
                value=attention_input
            )
            intra_attention_output = intra_attention_output.transpose(0, 1).mean(dim=1)  # (group_size, embedding_dim)

            # Transform intra-group attention output through linear layer
            attention_output = self.linear(intra_attention_output)
            combined_output[group_indices] = attention_output

            # Inter-group context aggregation using self-attention
            inter_group_outputs = []
            for other_group in unique_groups:
                if other_group != group:
                    other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
                    other_group_embeddings = normalized_embeddings[other_group_indices]

                    # Calculate inter-group interaction using self-attention
                    attention_output, _ = self.inter_attention_layer(
                        query=intra_attention_output.unsqueeze(1)[0],  # Using first in group
                        key=other_group_embeddings.unsqueeze(1)[0],
                        value=other_group_embeddings.unsqueeze(1)[0]
                    )
                    inter_group_outputs.append(attention_output.squeeze(1))

            if inter_group_outputs:
                # Average inter-group interaction results and combine with final output
                inter_group_output = torch.mean(torch.stack(inter_group_outputs), dim=0)
                combined_output[group_indices] += inter_group_output

        return combined_output


class Net(nn.Module):
    '''
    Neural Cognitive Diagnosis Model
    '''
    def __init__(self, student_n, exer_n, knowledge_n):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 11  # Configurable

        # New: theta bias embedding
        self.theta_bias = nn.Embedding(self.emb_num, 1, padding_idx=0)

        # Trainable beta and alpha parameters
        self.beta = nn.Parameter(torch.tensor(1.0))  # Initial value 1.0
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Initial value 1.0

        # Network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)  # Embedding for each student
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)  # Difficulty for each exercise's knowledge points
        self.e_discrimination = nn.Embedding(self.exer_n, 1)  # Discrimination for each exercise

        # Three fully connected layers corresponding to three interaction functions in the paper
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        self.prednet_full1_emb = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1_emb = nn.Dropout(p=0.5)
        self.prednet_full2_emb = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2_emb = nn.Dropout(p=0.5)
        self.prednet_full3_emb = nn.Linear(self.prednet_len2, 1)

        self.prednet_full1_xianyanfenbu = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1_xianyanfenbu = nn.Dropout(p=0.5)
        self.prednet_full2_xianyanfenbu = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2_xianyanfenbu = nn.Dropout(p=0.5)
        self.prednet_full3_xianyanfenbu = nn.Linear(self.prednet_len2, 1)

        # New: Linear layer to convert one-hot encoding to 10-dim vector (10 is number of knowledge points)
        self.linear_layer = nn.Linear(6, 11)

        # Initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, one_hot_input, group_labels):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, knowledge relevancy vectors (1 for relevant knowledge points, 0 otherwise)
        :return: FloatTensor, probabilities of answering correctly
        '''
        # Pre-prednet processing
        stu_emb = torch.sigmoid(self.student_emb(stu_id))  # Maps student ID to knowledge mastery level
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10  # Multiply by 10 to amplify discrimination impact

        # New: Influence weight coefficients
        con_belif = torch.sigmoid(self.theta_bias(stu_id)).view(-1, 1)  # 32 is batch size
        ablity_belif = torch.sigmoid(self.theta_bias(stu_id)).view(-1, 1)

        embeddings = self.linear_layer(one_hot_input)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)  # New: normalized embeddings

        # Use hierarchical information directly
        stu_xianyanfenbu_emb = torch.sigmoid(embeddings)

        # Prednet - core cognitive diagnosis + three interaction functions
        input_x1 = e_discrimination * (con_belif*stu_emb + (1-con_belif)*stu_xianyanfenbu_emb - k_difficulty) * kn_emb
        input_x1_emb = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x1_xianyanfenbu = e_discrimination * (stu_xianyanfenbu_emb - k_difficulty) * kn_emb

        input_x2 = self.drop_1(tanh_layer(self.prednet_full1(input_x1)))
        input_x3 = self.drop_2(tanh_layer(self.prednet_full2(input_x2)))
        output = torch.sigmoid(self.prednet_full3(input_x3))

        input_x2_emb = self.drop_1_emb(tanh_layer(self.prednet_full1_emb(input_x1_emb)))
        input_x3_emb = self.drop_2_emb(tanh_layer(self.prednet_full2_emb(input_x2_emb)))
        output_emb = torch.sigmoid(self.prednet_full3_emb(input_x3_emb))

        input_x2_xianyanfenbu = self.drop_1_xianyanfenbu(tanh_layer(self.prednet_full1_xianyanfenbu(input_x1_xianyanfenbu)))
        input_x3_xianyanfenbu = self.drop_2_xianyanfenbu(tanh_layer(self.prednet_full2_xianyanfenbu(input_x2_xianyanfenbu)))
        output_xianyanfenbu = torch.sigmoid(self.prednet_full3_xianyanfenbu(input_x3_xianyanfenbu))
        
        return output, output_emb, output_xianyanfenbu

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data

    # New: Calculate intra-group and inter-group loss
    def cosine_similarity_loss(self, embeddings, center):
        cos_sim = F.cosine_similarity(embeddings, center.expand_as(embeddings), dim=-1)
        return 1 - cos_sim.mean()

    def cosine_distance(self, center1, center2):
        return 1 - F.cosine_similarity(center1, center2, dim=0)

    def calculate_group_loss(self, embeddings, group_labels):
        group_loss = 0
        intra_group_loss_total = 0
        unique_groups = torch.unique(group_labels)

        for group in unique_groups:
            group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
            group_embeddings = embeddings[group_indices]
            group_center = group_embeddings.mean(dim=0)

            # Intra-group cosine similarity loss
            intra_group_loss = self.cosine_similarity_loss(group_embeddings, group_center)

            # Relaxed constraint loss
            relaxed_intra_loss = F.mse_loss(group_embeddings, group_center.expand_as(group_embeddings).detach())

            intra_group_loss_total += intra_group_loss + relaxed_intra_loss

            # Inter-group difference loss
            inter_group_loss = 0
            for other_group in unique_groups:
                if other_group != group:
                    other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
                    other_group_embeddings = embeddings[other_group_indices]
                    other_group_center = other_group_embeddings.mean(dim=0)
                    inter_group_loss += self.cosine_distance(group_center, other_group_center)

            group_loss += intra_group_loss_total

        return group_loss / len(unique_groups)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
