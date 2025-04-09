
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

# 设置随机种子
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

relu = torch.nn.ReLU()
# 创建 Tanh 层
tanh_layer = nn.Tanh()

#新增一个注意力机制 测试
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


##新增一个注意力机制，本文采用的该注意力机制
# class AttentionModel(nn.Module):
#     def __init__(self, embedding_dim):
#         super(AttentionModel, self).__init__()
#         self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=3)
#         self.multi_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=3)
#         self.linear = nn.Linear(embedding_dim, embedding_dim)
#
#     def forward(self, normalized_embeddings, group_labels):
#         batch_size, embedding_dim = normalized_embeddings.size()
#         combined_output = torch.zeros_like(normalized_embeddings)
#
#         unique_groups = torch.unique(group_labels)
#
#         for group in unique_groups:
#             group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
#             group_embeddings = normalized_embeddings[group_indices]
#
#             # 组内注意力  组内自己做注意力
#             attention_output, _ = self.attention_layer(
#                 query=group_embeddings.unsqueeze(1),
#                 key=group_embeddings.unsqueeze(1),
#                 value=group_embeddings.unsqueeze(1)
#             )
#             attention_output = attention_output.squeeze(1)
#
#             combined_output[group_indices] = attention_output
#
#             # 组间注意力  当前组与其他组分别做注意力
#             inter_group_outputs = []
#             for other_group in unique_groups:
#                 if other_group != group:
#                     other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
#                     other_group_embeddings = normalized_embeddings[other_group_indices]
#
#                     inter_attention_output, _ = self.multi_attention_layer(
#                         query=group_embeddings.unsqueeze(1),
#                         key=other_group_embeddings.unsqueeze(1),
#                         value=other_group_embeddings.unsqueeze(1)
#                     )
#                     inter_attention_output = inter_attention_output.squeeze(1)
#                     inter_group_outputs.append(inter_attention_output)
#
#             if inter_group_outputs:
#                 inter_group_output = torch.mean(torch.stack(inter_group_outputs), dim=0)
#                 combined_output[group_indices] += inter_group_output
#
#         return combined_output


# class AttentionModel(nn.Module):
#     def __init__(self, embedding_dim, conv_kernel_size=3):
#         super(AttentionModel, self).__init__()
#         self.conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
#         self.linear = nn.Linear(embedding_dim, embedding_dim)
#
#     def forward(self, normalized_embeddings, group_labels):
#         batch_size, embedding_dim = normalized_embeddings.size()
#         combined_output = torch.zeros_like(normalized_embeddings)
#
#         unique_groups = torch.unique(group_labels)
#
#         for group in unique_groups:
#             group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
#             group_embeddings = normalized_embeddings[group_indices]
#
#             # 通过卷积生成汇总tokens
#             # 汇总tokens表示多粒度上下文
#             aggregated_tokens = self.conv(group_embeddings.unsqueeze(0).transpose(1, 2))
#             aggregated_tokens = aggregated_tokens.transpose(1, 2).squeeze(0)
#
#             # 使用汇总tokens调制原始组内query
#             modulated_query = group_embeddings + aggregated_tokens
#
#             # 调制后的query传入线性层进一步变换
#             attention_output = self.linear(modulated_query)
#
#             combined_output[group_indices] = attention_output
#
#             # 组间上下文聚合: 将其他组的汇总tokens作为上下文信息
#             inter_group_outputs = []
#             for other_group in unique_groups:
#                 if other_group != group:
#                     other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
#                     other_group_embeddings = normalized_embeddings[other_group_indices]
#
#                     # 对其他组同样生成汇总tokens
#                     other_aggregated_tokens = self.conv(other_group_embeddings.unsqueeze(0).transpose(1, 2))
#                     other_aggregated_tokens = other_aggregated_tokens.transpose(1, 2).squeeze(0)
#
#                     # 计算组间交互，使用汇总tokens调制当前组
#                     # 将 other_aggregated_tokens 平均池化到 group_embeddings 的大小
#                     other_aggregated_tokens = other_aggregated_tokens.mean(dim=0, keepdim=True)
#                     other_aggregated_tokens = other_aggregated_tokens.expand_as(group_embeddings)
#                     inter_modulated_query = group_embeddings + other_aggregated_tokens
#
#                     inter_group_output = self.linear(inter_modulated_query)
#                     inter_group_outputs.append(inter_group_output)
#
#             if inter_group_outputs:
#                 # 将组间交互结果求平均，结合进最终输出
#                 inter_group_output = torch.mean(torch.stack(inter_group_outputs), dim=0)
#                 combined_output[group_indices] += inter_group_output
#
#         return combined_output

import torch
import torch.nn as nn

# class AttentionModel(nn.Module):
#     def __init__(self, embedding_dim, conv_kernel_size=3):
#         super(AttentionModel, self).__init__()
#         # 卷积层用于组内上下文聚合
#         self.conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
#         # 自注意力层用于组间上下文聚合
#         self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=3)
#         # 线性层用于最终输出
#         self.linear = nn.Linear(embedding_dim, embedding_dim)
#
#     def forward(self, normalized_embeddings, group_labels):
#         batch_size, embedding_dim = normalized_embeddings.size()
#         combined_output = torch.zeros_like(normalized_embeddings)
#
#         unique_groups = torch.unique(group_labels)
#
#         for group in unique_groups:
#             group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
#             group_embeddings = normalized_embeddings[group_indices]
#
#             # 通过卷积生成汇总tokens（组内上下文聚合）
#             aggregated_tokens = self.conv(group_embeddings.unsqueeze(0).transpose(1, 2))
#             aggregated_tokens = aggregated_tokens.transpose(1, 2).squeeze(0)
#
#             # 使用汇总tokens调制原始组内query
#             modulated_query = group_embeddings + aggregated_tokens
#
#             # 调制后的query传入线性层进一步变换
#             attention_output = self.linear(modulated_query)
#             combined_output[group_indices] = attention_output
#
#             # 组间上下文聚合: 使用自注意力机制
#             inter_group_outputs = []
#             for other_group in unique_groups:
#                 if other_group != group:
#                     other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
#                     other_group_embeddings = normalized_embeddings[other_group_indices]
#
#                     # 使用自注意力计算组间交互
#                     attention_output, _ = self.attention_layer(
#                         query=group_embeddings.unsqueeze(1),
#                         key=other_group_embeddings.unsqueeze(1),
#                         value=other_group_embeddings.unsqueeze(1)
#                     )
#                     inter_group_outputs.append(attention_output.squeeze(1))
#
#             if inter_group_outputs:
#                 # 将组间交互结果求平均，结合进最终输出
#                 inter_group_output = torch.mean(torch.stack(inter_group_outputs), dim=0)
#                 combined_output[group_indices] += inter_group_output
#
#         return combined_output

import torch
import torch.nn as nn


# class AttentionModel(nn.Module):
#     def __init__(self, embedding_dim, conv_kernel_size=3):
#         super(AttentionModel, self).__init__()
#         # 使用 2D 卷积，输入通道为 1，输出通道为 1，卷积核大小为 (conv_kernel_size, 1)
#         self.conv = nn.Conv2d(1, 1, kernel_size=(conv_kernel_size, 1), padding=(conv_kernel_size // 2, 0))
#
#         # 自注意力层用于组内上下文聚合
#         self.intra_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=3)
#
#         # 自注意力层用于组间上下文聚合
#         self.inter_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=3)
#
#         # 线性层用于最终输出
#         self.linear = nn.Linear(embedding_dim, embedding_dim)
#
#     def forward(self, normalized_embeddings, group_labels):
#         batch_size, embedding_dim = normalized_embeddings.size()
#         combined_output = torch.zeros_like(normalized_embeddings)
#
#         unique_groups = torch.unique(group_labels)
#
#         for group in unique_groups:
#             # 获取当前组的学生索引和嵌入
#             group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
#             group_embeddings = normalized_embeddings[group_indices]  # 形状为 (group_size, embedding_dim)
#
#             # 将组内嵌入组织为 (1, 1, group_size, embedding_dim) 以便2D卷积操作
#             group_embeddings = group_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, group_size, embedding_dim)
#
#             # 通过二维卷积生成汇总tokens（组内上下文聚合）
#             aggregated_tokens = self.conv(group_embeddings)  # 输出形状为 (1, 1, group_size, embedding_dim)
#             aggregated_tokens = aggregated_tokens.squeeze(0).squeeze(0)  # 形状为 (group_size, embedding_dim)
#
#             # 使用卷积后的表示进行组内的自注意力计算
#             # 需要转换形状为 (group_size, 1, embedding_dim) 来适应 MultiheadAttention 输入
#             attention_input = aggregated_tokens.unsqueeze(1)  # (group_size, 1, embedding_dim)
#             intra_attention_output, _ = self.intra_attention_layer(
#                 query=attention_input,
#                 key=attention_input,
#                 value=attention_input
#             )
#             intra_attention_output = intra_attention_output.squeeze(1)  # (group_size, embedding_dim)
#
#             # 将经过组内注意力后的表示传入线性层进一步变换
#             attention_output = self.linear(intra_attention_output)
#             combined_output[group_indices] = attention_output
#
#             # 组间上下文聚合: 使用自注意力机制
#             inter_group_outputs = []
#             for other_group in unique_groups:
#                 if other_group != group:
#                     other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
#                     other_group_embeddings = normalized_embeddings[other_group_indices]
#
#                     # 使用自注意力计算组间交互
#                     # 将组间的输入转换为 (group_size, 1, embedding_dim)
#                     attention_output, _ = self.inter_attention_layer(
#                         query=intra_attention_output.unsqueeze(1),
#                         key=other_group_embeddings.unsqueeze(1),
#                         value=other_group_embeddings.unsqueeze(1)
#                     )
#                     inter_group_outputs.append(attention_output.squeeze(1))
#
#             if inter_group_outputs:
#                 # 将组间交互结果求平均，结合进最终输出
#                 inter_group_output = torch.mean(torch.stack(inter_group_outputs), dim=0)
#                 combined_output[group_indices] += inter_group_output
#
#         return combined_output

import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, conv_kernel_sizes=[2, 4]):
        super(AttentionModel, self).__init__()

        # 为每个卷积核大小创建一个卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0)) for k in conv_kernel_sizes]
        )

        # 自注意力层用于组内上下文聚合
        self.intra_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=len(conv_kernel_sizes))

        # 自注意力层用于组间上下文聚合
        self.inter_attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=len(conv_kernel_sizes))

        # 线性层用于最终输出
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, normalized_embeddings, group_labels):
        batch_size, embedding_dim = normalized_embeddings.size()
        combined_output = torch.zeros_like(normalized_embeddings)

        unique_groups = torch.unique(group_labels)

        for group in unique_groups:
            # 获取当前组的学生索引和嵌入
            group_indices = (group_labels == group).nonzero(as_tuple=True)[0]
            group_embeddings = normalized_embeddings[group_indices]  # 形状为 (group_size, embedding_dim)
            group_size = group_embeddings.size(0)

            # 将组内嵌入组织为 (1, 1, group_size, embedding_dim) 以便2D卷积操作
            group_embeddings = group_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, group_size, embedding_dim)

            # 通过每个卷积层生成汇总tokens（组内上下文聚合），每个卷积层对应一个头
            aggregated_tokens = []
            for conv in self.convs:
                aggregated_token = conv(group_embeddings)  # 输出形状为 (1, 1, new_group_size, embedding_dim)
                aggregated_token = aggregated_token.squeeze(0).squeeze(0)  # 形状为 (new_group_size, embedding_dim)

                # 填充或裁剪到当前组的原始大小 group_size
                if aggregated_token.size(0) < group_size:
                    # 如果卷积后的长度较短，则在最后添加零以匹配原始大小
                    pad_size = group_size - aggregated_token.size(0)
                    aggregated_token = nn.functional.pad(aggregated_token, (0, 0, 0, pad_size))
                else:
                    # 如果卷积后的长度较长，则裁剪到原始大小
                    aggregated_token = aggregated_token[:group_size]

                aggregated_tokens.append(aggregated_token)

            # 将不同卷积层的输出堆叠并拼接，形状为 (group_size, num_heads, embedding_dim)
            aggregated_tokens = torch.stack(aggregated_tokens, dim=1)  # (group_size, num_heads, embedding_dim)

            # 使用卷积后的表示进行组内的自注意力计算
            # 需要转换形状为 (group_size, num_heads, embedding_dim) 来适应 MultiheadAttention 输入
            attention_input = aggregated_tokens.transpose(0, 1)  # (num_heads, group_size, embedding_dim)
            intra_attention_output, _ = self.intra_attention_layer(
                query=attention_input,
                key=attention_input,
                value=attention_input
            )
            intra_attention_output = intra_attention_output.transpose(0, 1).mean(dim=1)  # (group_size, embedding_dim)

            # 将经过组内注意力后的表示传入线性层进一步变换
            attention_output = self.linear(intra_attention_output)
            combined_output[group_indices] = attention_output

            # 组间上下文聚合: 使用自注意力机制
            inter_group_outputs = []
            for other_group in unique_groups:
                if other_group != group:
                    other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
                    # first_other_group_index = other_group_indices[0]  #随机选取第一个
                    other_group_embeddings = normalized_embeddings[other_group_indices]  #.unsqueeze(0)新增

                    # 使用自注意力计算组间交互
                    # 将组间的输入转换为 (group_size, 1, embedding_dim)
                    attention_output, _ = self.inter_attention_layer(
                        query=intra_attention_output.unsqueeze(1)[0],  #也是只要组内的第一个
                        key=other_group_embeddings.unsqueeze(1)[0],
                        value=other_group_embeddings.unsqueeze(1)[0]
                    )
                    inter_group_outputs.append(attention_output.squeeze(1))

            if inter_group_outputs:
                # 将组间交互结果求平均，结合进最终输出
                inter_group_output = torch.mean(torch.stack(inter_group_outputs), dim=0)
                combined_output[group_indices] += inter_group_output

        return combined_output


class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512,11  # changeable

        #新增
        self.theta_bias = nn.Embedding(self.emb_num, 1, padding_idx=0)

        # 定义可训练的 beta 参数
        self.beta = nn.Parameter(torch.tensor(1.0))  # 初始值为 1.0
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 初始值为 1.0

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)  # 给每一个学生进行嵌入
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)  # 给每个题目的每个知识点都进行难度嵌入
        self.e_discrimination = nn.Embedding(self.exer_n, 1)  # 给每个题目一个区分度嵌入

        # 这里的三个全连接层的预测 应该是对应论文里面的三个交互函数
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
        self.prednet_full3_xianyanfenbu= nn.Linear(self.prednet_len2, 1)

        #新增
        # 定义一个线性层，用于将 one-hot 编码转换为10 维向量 10是学生的知识点的个数
        self.linear_layer = nn.Linear(6,11)

        # self.attention_layer = CustomAttention(knowledge_n) #初始化注意力机制
        # self.AttentionModel_layer = AttentionModel(knowledge_n)


        # initialization  初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, one_hot_input,group_labels):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors  就是每个题目具体涉及了所有知识点中的哪些知识点 设计位置用1表示，不涉及的用0.0表示 注意 索引是从0开始
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))  # 每个学生唯一的编号，映射成了学生对知识点的熟练度（掌握程度）
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) *10  # 这里为什么要乘以11，我觉得只是为了扩大区分度的影响力 如果不乘以11  sigmod都是小于1的数 其实是缩小了区分度的影响力 和e_discrimination * (stu_emb - k_difficulty) 一起看？
        #新增
        # 影响权重系数
        con_belif = torch.sigmoid(self.theta_bias(stu_id)).view(-1, 1)  # 32是batchsize
        ablity_belif = torch.sigmoid(self.theta_bias(stu_id)).view(-1, 1)  # 32是batchsize

        embeddings = self.linear_layer(one_hot_input)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)  # 新增了归一化

        #使用注意力或者使用最简单直接的层次信息
        # combined_output = self.AttentionModel_layer(normalized_embeddings,group_labels)
        # stu_xianyanfenbu_emb = torch.sigmoid(combined_output)+ torch.sigmoid(embeddings)  # 加上原来的信息
        #层次信息
        stu_xianyanfenbu_emb = torch.sigmoid(embeddings)



        # prednet  认知诊断核心+三个交互函数
        input_x1 = e_discrimination * (con_belif*stu_emb + (1-con_belif)*stu_xianyanfenbu_emb  - k_difficulty) * kn_emb  # input_x相当于对当前题目的涉及知识点的把握度   训练完后的stu_emb相当于XX学生的能力值（在每个知识点的能力值，或熟练程度）
        # input_x1 = e_discrimination * ( stu_xianyanfenbu_emb - k_difficulty) * kn_emb
        #新增
        input_x1_emb = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x1_xianyanfenbu = e_discrimination * (stu_xianyanfenbu_emb - k_difficulty) * kn_emb

        input_x2 = self.drop_1(tanh_layer(self.prednet_full1(input_x1)))
        input_x3 = self.drop_2(tanh_layer(self.prednet_full2(input_x2)))
        # input_x3 = input_x1 + input_x3  #我自己的加的 飞哥的没有
        output = torch.sigmoid(self.prednet_full3(input_x3))

        input_x2_emb = self.drop_1_emb(tanh_layer(self.prednet_full1_emb(input_x1_emb)))
        input_x3_emb = self.drop_2_emb(tanh_layer(self.prednet_full2_emb(input_x2_emb)))
        output_emb = torch.sigmoid(self.prednet_full3_emb(input_x3_emb))

        input_x2_xianyanfenbu= self.drop_1_xianyanfenbu(tanh_layer(self.prednet_full1_xianyanfenbu(input_x1_xianyanfenbu)))
        input_x3_xianyanfenbu = self.drop_2_xianyanfenbu(tanh_layer(self.prednet_full2_xianyanfenbu(input_x2_xianyanfenbu)))
        output_xianyanfenbu = torch.sigmoid(self.prednet_full3_xianyanfenbu(input_x3_xianyanfenbu))
        # return output
        return output,output_emb,output_xianyanfenbu

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



    #新增计算组内和组间loss

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

            # 组内余弦相似度损失
            intra_group_loss = self.cosine_similarity_loss(group_embeddings, group_center)

            # 缓和约束的损失
            relaxed_intra_loss = F.mse_loss(group_embeddings, group_center.expand_as(group_embeddings).detach())

            intra_group_loss_total += intra_group_loss + relaxed_intra_loss

            #组间差异性损失
            inter_group_loss = 0
            for other_group in unique_groups:
                if other_group != group:
                    other_group_indices = (group_labels == other_group).nonzero(as_tuple=True)[0]
                    other_group_embeddings = embeddings[other_group_indices]
                    other_group_center = other_group_embeddings.mean(dim=0)
                    inter_group_loss += self.cosine_distance(group_center, other_group_center)  # 累加组间损失

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
