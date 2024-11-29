import torch
import torch.nn as nn
import math

from torch.nn.utils import weight_norm
from util.env import get_device

from models_ver3.graph_layer import GraphLayer


def get_batch_edge_index2(org_edge_index, batch_num, node_num):
    device = get_device()

    edge_index = org_edge_index.clone()
    batch_edge_index = edge_index.repeat(1, batch_num)
    inc = torch.arange(batch_num).repeat_interleave(edge_index.shape[1]) * node_num

    batch_edge_index = batch_edge_index + inc.to(device)

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, layer_num, inter_num=512, dropout=0.2):
        super(OutLayer, self).__init__()

        self.mlp = nn.ModuleList()

        for i in range(layer_num):
            if i == layer_num-1:
                self.mlp.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                self.mlp.append(nn.Linear(layer_in_num, inter_num))
                # self.mlp.append(nn.BatchNorm1d(inter_num))
                self.mlp.append(nn.ReLU())
                self.mlp.append(nn.Dropout(p=dropout))

        self.reset_parameters()

    def reset_parameters(self):
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.constant_(mod.weight, 1.0)
                nn.init.constant_(mod.bias, 0.0)
            elif isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0.0)
            else:
                pass

    def forward(self, x):
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                x = x.permute(0, 2, 1)
                x = mod(x)
                x = x.permute(0, 2, 1)
            else:
                x = mod(x)

        return x


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, d_embedding, dropout):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, d_embedding, dropout=dropout)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

        self.att_weight_1 = None
        self.edge_index_1 = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x, edge_index, embedding):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)

        self.edge_index_1 = new_edge_index
        self.att_weight_1 = att_weight

        #out = self.bn(out)
        
        return self.relu(out)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TCN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.conv1 = weight_norm(nn.Conv1d(self.input_size, self.output_size, self.kernel_size,
                                           padding=(kernel_size-1)*1, dilation=1))
        self.chomp1 = Chomp1d(kernel_size-1)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(self.output_size, self.output_size, self.kernel_size,
                                           padding=(kernel_size-1)*2, dilation=2))
        self.chomp2 = Chomp1d((kernel_size-1)*2)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout)

        self.conv3 = weight_norm(nn.Conv1d(self.output_size, self.output_size, self.kernel_size,
                                           padding=(kernel_size-1)*4, dilation=4))
        self.chomp3 = Chomp1d((kernel_size-1)*4)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.conv1.bias.data.fill_(0.0)
        self.conv2.bias.data.fill_(0.0)
        self.conv3.bias.data.fill_(0.0)

    def forward(self, x):
        z1 = self.conv1(x)
        z1 = self.chomp1(z1)
        z1 = self.relu1(z1)
        z1 = self.dp1(z1)
        # z1 = self.relu(z1 + x)

        z2 = self.conv2(z1)
        z2 = self.chomp2(z2)
        z2 = self.relu2(z2)
        z2 = self.dp2(z2)
        # z2 = self.relu(z2 + z1)

        z3 = self.conv3(z2)
        z3 = self.chomp3(z3)
        z3 = self.relu3(z3)
        z3 = self.dp3(z3)
        # z3 = self.relu(z3 + z2)
        z3 = self.relu(z3 + x)

        return z3


class PF(nn.Module):
    def __init__(self,
                 edge_index,
                 node_num,
                 process_info,
                 mask1,
                 mask2,
                 emb_dim=64,
                 feature_dim=64,
                 slide_win=10,
                 out_layer_num=1,
                 out_layer_inter_dim=256,
                 topk=20,
                 alpha=0.3,
                 dropout=0.2,
                 out_dropout=0.2,
                 kernel_size=3,
                 out_mode=1,
                 ):
        super(PF, self).__init__()

        self.device = get_device()

        self.edge_index = edge_index
        self.node_num = node_num
        self.process_info = process_info
        self.mask1 = mask1
        self.mask2 = mask2

        self.slide_win = slide_win
        self.topk = topk
        self.alpha = alpha

        self.topk_list1 = [round(alpha*len([j for j in self.mask1[:, i] if j == 1])) for i in range(node_num)]
        self.topk_list2 = [round(alpha*len([j for j in self.mask2[:, i] if j == 1])) for i in range(node_num)]

        self.s_emb = nn.Embedding(node_num, emb_dim)
        self.t_emb = nn.Embedding(node_num, emb_dim)

        self.TCN = TCN(self.node_num, self.node_num, kernel_size=kernel_size, dropout=dropout)

        self.gnn_layer1 = GNNLayer(slide_win, feature_dim, emb_dim, dropout)
        self.gnn_layer2 = GNNLayer(slide_win, feature_dim, emb_dim, dropout)
        self.gnn_layer3 = GNNLayer(slide_win, feature_dim, emb_dim, dropout)

        self.relu = nn.ReLU()
        self.adj_ui = None
        self.adj_sg = None
        self.adj_pf = None

        self.out_mode = out_mode
        print(f'out_mode: {out_mode}')

        if out_mode == 1:
            out_in_dim = 2*feature_dim+2*emb_dim
        elif out_mode == 2:
            out_in_dim = feature_dim+2*emb_dim
        elif out_mode == 3:
            out_in_dim = feature_dim+emb_dim
        elif out_mode == 4:
            assert feature_dim == emb_dim
            out_in_dim = feature_dim
        else:
            assert out_mode == 5
            assert feature_dim == emb_dim
            out_in_dim = 2*feature_dim

        self.out_layer = OutLayer(out_in_dim,
                                  layer_num=out_layer_num,
                                  inter_num=out_layer_inter_dim,
                                  dropout=out_dropout)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.s_emb.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.t_emb.weight, a=math.sqrt(5))

    def forward(self, x):
        batch_num, node_num, slide_win = x.shape
        assert node_num == self.node_num
        assert slide_win == self.slide_win

        z = self.TCN(x).view(-1, slide_win)

        s_emb = self.s_emb(torch.arange(node_num).to(self.device))
        t_emb = self.t_emb(torch.arange(node_num).to(self.device))

        batch_s_emb = s_emb.repeat(batch_num, 1)
        batch_t_emb = t_emb.repeat(batch_num, 1)

        ns_emb = s_emb / s_emb.norm(dim=-1).unsqueeze(-1)
        nt_emb = t_emb / t_emb.norm(dim=-1).unsqueeze(-1)

        c_ji = self.relu(torch.matmul(ns_emb, nt_emb.t())).detach().clone()
        c_ji.fill_diagonal_(float('-inf'))

        # uninformed
        topk_indices = torch.topk(c_ji, self.topk, dim=0)[1]

        self.adj_ui = topk_indices

        gated_i = torch.arange(node_num).repeat_interleave(self.topk).unsqueeze(0).to(self.device)
        gated_j = topk_indices.t().contiguous().flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

        batch_gated_edge_index = get_batch_edge_index2(gated_edge_index, batch_num, node_num)
        gnn_out1 = self.gnn_layer1(z, batch_gated_edge_index, embedding=[batch_s_emb, batch_t_emb])

        # masking
        c_ji_sg = torch.where(self.mask1 == 0, torch.tensor(float('-inf')).to(self.device), c_ji)
        c_ji_pf = torch.where(self.mask2 == 0, torch.tensor(float('-inf')).to(self.device), c_ji)

        topk_indices_sg = []
        topk_indices_pf = []

        gated_sg_i = []
        gated_sg_j = []

        gated_pf_i = []
        gated_pf_j = []

        for i in range(self.node_num):
            #topk_indices_sg_i = torch.topk(c_ji_sg[:, i], self.topk_list1[i])[1].tolist()
            #topk_indices_sg.append(topk_indices_sg_i)

            #gated_sg_i.extend([i] * len(topk_indices_sg_i))
            #gated_sg_j.extend(topk_indices_sg_i)

            topk_indices_pf_i = torch.topk(c_ji_pf[:, i], self.topk_list2[i])[1].tolist()
            topk_indices_pf.append(topk_indices_pf_i)

            gated_pf_i.extend([i] * len(topk_indices_pf_i))
            gated_pf_j.extend(topk_indices_pf_i)

        # sensor group
        '''self.adj_sg = topk_indices_sg

        gated_sg_i = torch.tensor(gated_sg_i).unsqueeze(0).to(self.device)
        gated_sg_j = torch.tensor(gated_sg_j).unsqueeze(0).to(self.device)
        gated_edge_index_sg = torch.cat((gated_sg_j, gated_sg_i), dim=0)

        batch_gated_edge_index_sg = get_batch_edge_index2(gated_edge_index_sg, batch_num, node_num)
        gnn_out2 = self.gnn_layer2(z, batch_gated_edge_index_sg, embedding=[batch_s_emb, batch_t_emb])'''

        # process flow
        self.adj_pf = topk_indices_pf

        gated_pf_i = torch.tensor(gated_pf_i).unsqueeze(0).to(self.device)
        gated_pf_j = torch.tensor(gated_pf_j).unsqueeze(0).to(self.device)
        gated_edge_index_pf = torch.cat((gated_pf_j, gated_pf_i), dim=0)

        batch_gated_edge_index_pf = get_batch_edge_index2(gated_edge_index_pf, batch_num, node_num)
        gnn_out3 = self.gnn_layer3(z, batch_gated_edge_index_pf, embedding=[batch_s_emb, batch_t_emb])

        # output
        if self.out_mode == 1:
            h_f = torch.cat((gnn_out1, gnn_out3,
                             batch_s_emb, batch_t_emb), dim=-1).view(batch_num, node_num, -1)
        elif self.out_mode == 2:
            h_f = torch.cat((gnn_out1 + gnn_out3,
                             batch_s_emb, batch_t_emb), dim=-1).view(batch_num, node_num, -1)
        elif self.out_mode == 3:
            h_f = torch.cat((gnn_out1 + gnn_out3,
                             batch_s_emb + batch_t_emb), dim=-1).view(batch_num, node_num, -1)
        elif self.out_mode == 4:
            h_f = torch.mul(gnn_out1 + gnn_out3,
                            batch_s_emb + batch_t_emb).view(batch_num, node_num, -1)
        else:
            gnn_out_sum = gnn_out1 + gnn_out3
            h_f = torch.cat((torch.mul(gnn_out_sum, batch_s_emb),
                             torch.mul(gnn_out_sum, batch_t_emb)), dim=-1).view(batch_num, node_num, -1)

        out_f = self.out_layer(h_f)

        out_f = out_f.squeeze(-1)

        return out_f
