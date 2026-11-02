
import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch_sparse import SparseTensor, matmul
# from torch_geometric.utils import degree
import torch
import sys


def attention_adj(qs, ks):
    # qs = qs / torch.norm(qs, p=2) # [N, H, M]
    # ks = ks / torch.norm(ks, p=2) # [L, H, M]
    # numerator
    attention_num = torch.sigmoid(torch.einsum("ij,kj->ik", qs, ks))  # [N, L, H]\ qkt
    # denominator   sqrt（dk）
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    attention_normalizer = torch.einsum("nl,l->n", attention_num, all_ones)
    attention_normalizer = attention_normalizer.unsqueeze(0).repeat(ks.shape[0], 1)
    # compute attention and attentive aggregated results
    attention = attention_num / attention_normalizer

    return attention

def get_intra_value(attention, vs):
    attn_output = torch.einsum("ij,jk->ik", attention, vs)
    return attn_output

def get_inter_value(P, V):
    final = torch.einsum('ij,ivd->jvd', P, V)
    return final

def full_attention_conv(qs, ks, vs, output_attn=False):
    # qs = qs / torch.norm(qs, p=2) # [N, H, M]
    # ks = ks / torch.norm(ks, p=2) # [L, H, M]
    # numerator
    attention_num = torch.sigmoid(torch.einsum("ij,kj->ik", qs, ks))  # [N, L, H]\ qkt

    # denominator   sqrt（dk）
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    attention_normalizer = torch.einsum("nl,l->n", attention_num, all_ones)
    attention_normalizer = attention_normalizer.unsqueeze(0).repeat(ks.shape[0], 1)

    # compute attention and attentive aggregated results
    # attention = attention_num / attention_normalizer
    attention = attention_num
    attn_output = torch.einsum("ij,jk->ik", attention, vs)  # [N, H, D]
    if output_attn:
        return attn_output, attention_num
    else:
        return attn_output
    
def sgformer_attention_conv(qs, ks, vs, output_attn=False):
# def full_attention_conv(qs, ks, output_attn=False):
    qs = F.relu(qs / torch.norm(qs, p=2))  # [N, H, M]
    ks = F.relu(ks / torch.norm(ks, p=2))  # [L, H, M]
    qs = qs / torch.norm(qs, p=2) # [N, H, M]
    ks = ks / torch.norm(ks, p=2) # [L, H, M]
    N = qs.shape[0]


    kv = torch.einsum("ij,ik->jk", ks, vs)

    # denominator   sqrt（dk）
    all_ones = torch.ones([qs.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("nl,n->l", ks, all_ones)

    attention_normalizer = torch.einsum("nl,l->n", qs, ks_sum) + N * all_ones
    attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[1])


    attn_output = N * torch.einsum("ij,jk->ik", qs,kv) /(attention_normalizer+1e-9)

    return attn_output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, batch_norm=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.tanh = nn.Tanh()  # tanh激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        self.dropout = dropout
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        if self.batch_norm:   # 加了不好
            out = self.bn1(out)
        out = self.tanh(out)
        out = F.dropout(out, self.dropout, training=self.training)  # 加了更好
        out = self.fc2(out)
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()




class IntraDiffCov(nn.Module):
    def __init__(self, in_channels, out_channels, views):
        super(IntraDiffCov, self).__init__()
        self.Wklist = nn.ModuleList()
        self.Wqlist = nn.ModuleList()
        self.Wvlist = nn.ModuleList()
        for v in range(views):
            self.Wklist.append(nn.Linear(in_channels, out_channels))
            self.Wqlist.append(nn.Linear(in_channels, out_channels))
            self.Wvlist.append(nn.Linear(in_channels, out_channels))
        self.out_channels = out_channels
        self.views = views
    def reset_parameters(self):
        for Wk in self.Wklist:
            Wk.reset_parameters()
        for Wq in self.Wqlist:
            Wq.reset_parameters()


    def forward(self, qk, Value, mask_in, mask_out, mask_iso, v):
        attention_output = []
        latent = qk
        Key = self.Wklist[v](latent)
        Query = self.Wqlist[v](latent)
        mask_in_expanded = ~mask_in.unsqueeze(1).expand(-1, Key.size(1))
        mask_out_expanded = ~mask_out.unsqueeze(1).expand(-1, Key.size(1))
        mask_iso_expanded = ~mask_iso.unsqueeze(1).expand(-1, Key.size(1))
        # mask_in_expanded = ~mask_in[None, :].transpose(0, 1).expand_as(Key)
        # mask_out_expanded = ~mask_out[None, :].transpose(0, 1).expand_as(Query)
        # mask_iso_expanded = ~mask_iso[None, :].transpose(0, 1).expand_as(Query)

        mask_in_expanded = mask_in_expanded & mask_iso_expanded
        mask_out_expanded = mask_out_expanded & mask_iso_expanded

        Key = Key * mask_in_expanded.float()
        Query = Query * mask_out_expanded.float()




        # Value = latent_feature[v,:,:]
        attention_view = sgformer_attention_conv(Query, Key, Value) 
        # attention_output.append(attention_view)
        # attention_output = full_attention_conv(Query,Key)# [N, H, D]
            # intra_view.append(attention_output)
            # attention_view.append(attention)
        # intra_view = torch.stack(intra_view, dim=0)
        # attention_output = torch.stack(attention_output, dim=0)

        return attention_view




# class IntraDiffCov(nn.Module):
#     def __init__(self, in_channels, out_channels, view):
#         super(IntraDiffCov, self).__init__()
#         self.Wklist = nn.ModuleList()
#         self.Wqlist = nn.ModuleList()
#         for v in range(view):
#             self.Wklist.append(nn.Linear(in_channels, out_channels))
#             self.Wqlist.append(nn.Linear(in_channels, out_channels))
#         self.out_channels = out_channels
#         self.views = view
#     def reset_parameters(self):
#         for Wk in self.Wklist:
#             Wk.reset_parameters()
#         for Wq in self.Wqlist:
#             Wq.reset_parameters()
#
#     def forward(self, latent_feature):
#         # feature transformation
#         attention_adj_list = []
#
#         for v in range(self.views):
#             Key = self.Wklist[v](latent_feature[v, :, :])
#             Query = self.Wqlist[v](latent_feature[v, :, :])
#             attention = attention_adj(Key, Query)
#             attention_adj_list.append(attention)
#
#         # for v in range(self.views):
#         #     Key = self.Wklist[v](latent_feature[v,:,:])
#         #     Query = self.Wqlist[v](latent_feature[v,:,:])
#         #     Value = self.Wvlist[v](latent_feature[v,:,:])
#         #     attention_output = full_attention_conv(Query,Key,Value) # [N, H, D]
#         #     intra_view.append(attention_output)
#             # attention_view.append(attention)
#         attention_adj_list = torch.stack(attention_adj_list, dim=0)
#         return attention_adj_list

class InterDiffCov(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(InterDiffCov, self).__init__()
        self.args = args
        self.WQ = nn.Linear(in_channels, out_channels)
        self.WK = nn.Linear(in_channels, out_channels)
        self.WV = nn.Linear(in_channels, out_channels)

    def projection_p(self, P):
        dim = P.shape[0]
        if self.args.dataset in ['ACM','DBLP','IMDB','amazon','ALIBABA','YELP']:
            device = str(self.args.device)
        else:
            device ='cuda:' + self.args.device
        one = torch.ones(dim, 1).to(device)

        relu = torch.nn.ReLU()
        '''
        Projection
        '''
        P_1 = relu(P)

        support_1 = torch.mm(torch.mm(P_1, one) - one, one.t()) / dim
        P_2 = P_1 - support_1

        support_2 = torch.mm(one, torch.mm(one.t(), P_2) - one.t()) / dim
        P_3 = P_2 - support_2

        return P_3

    def softmax_projection(self, x):
        sm_0 = torch.nn.Softmax(dim=0)
        sm_1 = torch.nn.Softmax(dim=1)

        x_0 = sm_0(x)
        x_1 = sm_1(x)

        proj_x = (x_0 + x_1) / 2
        return proj_x

    def multi_projection(self, x):
        proj_x = self.softmax_projection(x)
        for i in range(10):
            proj_x = self.projection_p(proj_x)

        return proj_x

    def forward(self, latent_feature):
        Q = self.WQ(latent_feature)
        K = self.WK(latent_feature)
        Q = Q / torch.norm(Q, p='fro', dim=(1, 2), keepdim=True)
        K = K / torch.norm(Q, p='fro', dim=(1, 2), keepdim=True)

        P = torch.sigmoid(torch.einsum('ivd,jvd->ij', Q, K))
        # P = self.multi_projection(P)
        V = latent_feature

        final = torch.einsum('ij,ivd->jvd', P, V)

        return final

class CrossFormer(nn.Module):
    '''
    CrossFormer model class
    x: input node features [N, Dv]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels_list, hidden_channels, out_channels, view, args):
        super(CrossFormer, self).__init__()
        self.view = len(in_channels_list)
        self.Intraconvsin = nn.ModuleList()
        self.Intraconvsout = nn.ModuleList()
        self.Intra = nn.ModuleList()
        self.Intercovsin = nn.ModuleList()
        self.Intercovsout = nn.ModuleList()
        self.Wvlist = nn.ModuleList()
        # self.WVlist = nn.ModuleList()
        self.maplist = nn.ModuleList()
        self.envirin = nn.ModuleList()
        self.envirout = nn.ModuleList()
        self.edgeintramapoutlist = nn.ModuleList()

        self.sgformer = nn.ModuleList()
        self.Intraconvs = nn.ModuleList()
        self.Intercovs = nn.ModuleList()
        # encoder = MLP()

        self.fcs = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.args = args
        for v in range(self.view):
            # self.fcs.append(MLP(in_channels_list[v], hidden_channels * 2,hidden_channels))
            self.fcs.append(nn.Linear(in_channels_list[v], hidden_channels))
            self.Wvlist.append(nn.Linear(hidden_channels, hidden_channels))
            # dim = [hidden_channels,hidden_channels//2,hidden_channels//4,4] #flickr webkb cal
            dim = [hidden_channels, hidden_channels // 2, 4]  # BDGP out HW n noisy
            edgeintramapin = nn.ModuleList()
            for i in range(len(dim)-1):
                edgeintramapin.append(nn.Linear(dim[i], dim[i+1]))
            self.envirin.append(edgeintramapin)

        self.alpha = args.alpha
        self.layers = args.layers
        for i in range(args.layers):
            WVlist = nn.ModuleList()
            for j in range(self.view):
                WVlist.append(nn.Linear(hidden_channels, hidden_channels))
            self.maplist.append(WVlist)
            self.Intraconvsin.append(IntraDiffCov(hidden_channels, hidden_channels, view))
            self.Intercovsin.append(InterDiffCov(args, hidden_channels, hidden_channels))
            self.Intercovsout.append(InterDiffCov(args, hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = args.dropout
        self.activation = F.relu


    def reset_parameters(self):
        for conv in self.Intraconvs:
            conv.reset_parameters()
        for conv in self.Interconvs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()



    def forward(self, x_list):
        # input MLP
        args = self.args
        x_latent = []
        history_score = []
        history_action = []
        for v in range(self.view):
            x_latent.append(self.fcs[v](x_list[v]))
            history_score.append(torch.randn(x_list[v].size(0), 1))
            history_action.append(torch.rand(x_list[v].size(0), 1) > 0.5)
        x_latent = torch.stack(x_latent, dim=0)
        loss_re = 0
        # history = torch.zeros((x_list[v].size(0), 1), dtype=torch.bool).cuda(0)
        for i in range(self.layers):
        
            X_ins = []
            for v in range(self.view):
                intra_Value = self.maplist[i][v](x_latent[v, :, :])
                in_logits = intra_Value

                for en_in in self.envirin[v]:
                    in_logits = F.sigmoid(en_in(in_logits))

                eta = args.eta
                if i != 0:
                    in_logits[:, 3] = in_logits[:, 3] + eta * in_logits[:, 3] * history_score[v] * history_action[v]

                history_score[v] = in_logits[:, 3]
                temp = args.temp
                in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
                # keep_stand_probs = in_probs[:, 0].reshape(-1, 1)
                keep_in_probs = in_probs[:, 1].reshape(-1, 1)
                keep_out_probs = in_probs[:, 2].reshape(-1, 1)
                keep_iso_probs = in_probs[:, 3].reshape(-1, 1)

                # mask_stand = keep_stand_probs.bool().squeeze()  # 不做mask处理
                mask_in = keep_in_probs.bool().squeeze()  # mask 列
                mask_out = keep_out_probs.bool().squeeze()  # mask 行
                mask_iso = keep_iso_probs.bool().squeeze()  # mask 行列
                # if i != 0:
                #     mask_iso = history_action[v] | mask_iso
                history_action[v] = mask_iso
                x_intra_in = self.Intraconvsin[i](x_latent[v, :, :], intra_Value, mask_in, mask_out, mask_iso, v) + intra_Value
                loss_re += torch.norm((intra_Value - x_intra_in), p='fro')

                X_ins.append(x_intra_in)


            x_latent = torch.stack(X_ins, dim=0)
            X_ins_s = self.Intercovsin[i](x_latent)
            # X_out_s = self.Intercovsout[i](x_latent)
            delta = args.delta
            x_latent = delta * X_ins_s + (1 - delta) * x_latent


        loss_all = loss_re/self.view
        final = torch.sum(x_latent, dim=0)
        x_out = self.fc(final)

        return x_out, loss_all

















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def kl_loss_function(mu, logvar):
#     kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return kl
# class VAE(nn.Module):
#     def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims, z_dim,
#                  use_batchnorm=True):
#         super(VAE, self).__init__()
#         self.use_batchnorm = use_batchnorm
#         # Encoder 网络
#         encoder_layers = []
#         prev_dim = input_dim
#         for hidden_dim in encoder_hidden_dims:
#             encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
#             # if self.use_batchnorm:
#             #     encoder_layers.append(nn.BatchNorm1d(hidden_dim))
#             encoder_layers.append(nn.LeakyReLU())
#             prev_dim = hidden_dim
#         # encoder_layers.append(nn.Linear(prev_dim, z_dim))
#         self.encoder = nn.Sequential(*encoder_layers)
#
#         self.batch_norm = nn.BatchNorm1d(prev_dim)
#
#         self.fc_mu = nn.Linear(prev_dim, z_dim)
#         self.fc_logvar = nn.Linear(prev_dim, z_dim)
#
#         # Decoder 网络
#         decoder_layers = []
#         prev_dim = z_dim   # 输入为 z_dim
#         for hidden_dim in decoder_hidden_dims:  # 对于解码器，逆向排列层
#             decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
#             # if self.use_batchnorm:
#             #     decoder_layers.append(nn.BatchNorm1d(hidden_dim))
#             decoder_layers.append(nn.LeakyReLU())
#             prev_dim = hidden_dim
#         decoder_layers.append(nn.Linear(prev_dim, input_dim))  # 输出层
#         self.decoder = nn.Sequential(*decoder_layers)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     # 假设先验为标准正态
#
#
#     def forward(self, x):
#         hidden = self.encoder(x)
#         hb = self.batch_norm(hidden)
#         mu = self.fc_mu(hb)
#         logvar = self.fc_logvar(hidden)
#         z = F.dropout(self.reparameterize(mu, logvar))
#         x_hat = self.decoder(z)
#         # 假设先验为标准正态
#         re_loss = 0
#         kl_loss = kl_loss_function(mu, logvar)
#         # re_loss = F.mse_loss(x, x_hat, reduction='sum')
#         # return z, mu, re_loss, kl_loss
#
#         return z, re_loss, kl_loss
#
