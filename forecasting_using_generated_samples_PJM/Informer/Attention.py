import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries,
                              keys)  # query:decoder_input 与 encoder_outputb,h,seq_len,start_len
        #        print(scores.shape)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 取scale
        V = torch.einsum("bhls,bshd->blhd", A, values)
        #        print(V.shape)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # 先增加一个维度，相当于复制，再扩充
        #        print(K_expand.shape)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q 构建96*25的随机数
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # 随机取出的25个k值
        #        print(K_sample.shape)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # 96个Q和25个K之间的关系
        #        print(Q_K_sample.shape)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # 96个Q中每一个选跟其他K关系最大的值 再计算与均匀分布的差异
        #        print(Q_K_sample.max(-1)[0].shape)
        #        print(M.shape)
        M_top = M.topk(n_top, sorted=False)[1]  # 对96个Q的评分中选出25个 返回值1表示要得到索引
        #        print(M_top.shape)

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q) 取出来Q的特征
        #        print(Q_reduce.shape)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k 25个Q和全部K之间的关系
        #        print(Q_K.shape)
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            #            print(V_sum.shape)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # 先把96个V都用均值来替换
        #            print(contex.shape)
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)  # 累加
        #            print(contex.shape)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q,
                        attn_mask):  # context:初始值，一般初始是每个序列的均值，v为value，score表示top的分数，index表示top的index，l_q是序列长度
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)  # mask中TRUE的部分填充为inf
        #            print(scores.shape)
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        #        print(attn.shape)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 对25个有Q的更新V，其余的没变还是均值
        #        print(context_in.shape)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            #            print(attns.shape)
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        '''
        注意力改编
        :param queries:B*seq_len*head*(d_model/head)
        :param keys: B*seq_len*head*(d_model/head)
        :param values: B*seq_len*head*(d_model/head)
        :param attn_mask:
        :return: 返回attention矩阵
        '''
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # (32,72,8,64)——>(32,8,72,64)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) Key里要选的个数
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)  # 得到分数最高的u个，返回u个的index

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context，刚开始将所有的value赋值为每个序列的均值（48个和的均值）
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q,
                                             attn_mask)  # 根据score和top-index更新value值

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # (32,72,512)——>(32,72,8,64)
        keys = self.key_projection(keys).view(B, S, H, -1)  # (32,72,512)——>(32,72,8,64)
        values = self.value_projection(values).view(B, S, H, -1)  # (32,72,512)——>(32,72,8,64)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)  # B,seq_len,d_model

        return self.out_projection(out), attn

