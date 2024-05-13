import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):  

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


def CausalConvolution(input_size,output_size,kernel_size,dilation):
    return nn.Conv1d(input_size,output_size,kernel_size,padding=(kernel_size-1)*dilation,dilation=dilation)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d,self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self,x):
        return x[:,:,:-self.chomp_size].contiguous()
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size = 3,dropout=0.2):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dropout = 0.2
        self.conv1 = weight_norm(CausalConvolution(self.input_size,self.output_size,self.kernel_size,dilation=1))
        self.chomp1 = Chomp1d(self.kernel_size-1)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(self.dropout)
        self.conv2 = weight_norm(CausalConvolution(self.input_size,self.output_size,self.kernel_size,dilation=2))
        self.chomp2 = Chomp1d((self.kernel_size-1)*2)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(self.dropout)
        self.conv3 = weight_norm(CausalConvolution(self.input_size,self.output_size,self.kernel_size,dilation=4))
        self.chomp3 = Chomp1d((self.kernel_size-1)*4)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.init_weights()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dp1,
                                 self.conv2, self.chomp2, self.relu2, self.dp2
                                 ,self.conv3, self.chomp3, self.relu3, self.dp3
                                 )
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self,x):
        out = self.net(x)
        res = x
        return self.relu(out + res)
    
class Proposed(nn.Module):
    def __init__(self, edge_index_sets, node_num,process_info, mask1,mask2, sensor_dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, 
                 topk=20,alpha=0.3):

        super(Proposed, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()
        self.node_num = node_num
        edge_index = edge_index_sets[0]
        self.temp_dim = input_dim
        self.process_info = process_info  
        self.mask1 = mask1 
        self.mask2 = mask2 
        self.process_num = len(process_info) 
        self.s_embedding = nn.Embedding(node_num, sensor_dim).to(device) 
        self.s_embedding2 = nn.Embedding(node_num, sensor_dim).to(device) 
        self.embedding_list = [self.s_embedding,self.s_embedding2]
        dim = sensor_dim
        self.TCN = TCN(self.node_num, self.node_num, kernel_size=3, dropout=0.2) 
        embed_dim = dim
        total_dim = dim+sensor_dim+sensor_dim
        
        self.bn_outlayer_in = nn.BatchNorm1d(total_dim) 
        self.alpha = alpha
        self.topk_list = [topk]*node_num 
        self.topk_list1 = [round(alpha*len([j for j in self.mask1[i] if j!=0])) for i in range(node_num)] 
        self.topk_list2 = [round(alpha*len([j for j in self.mask2[i] if j!=0])) for i in range(node_num)] 

        edge_set_num = len(edge_index_sets)  #1
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.temp_dim, dim, inter_dim=dim+embed_dim, heads=1),GNNLayer(self.temp_dim, sensor_dim, inter_dim=dim+embed_dim, heads=1)
        ]) 

        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1) 
        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(total_dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim) ##total_embedding dim+sensor embedding dim -> 1

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    
    def init_params(self):
        for embeddings in self.embedding_list:
            nn.init.kaiming_uniform_(embeddings.weight, a=math.sqrt(5)) 


    def forward(self, data, org_edge_index):
        
        x = data.clone().detach()
        
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = self.TCN(x).view(-1,all_feature).contiguous().to(device) 
        gcn_outs = []
        
        topk_num = self.topk
        
        for i, edge_index in enumerate(edge_index_sets):  #0, fc_edge_index
            edge_num = edge_index.shape[1]                #27x26
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)  #(2,32x27x26)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            s_emb = self.s_embedding(torch.arange(node_num).to(device)).to(device)
            s_emb2 = self.s_embedding2(torch.arange(node_num).to(device)).to(device)
            embeddings1 = s_emb
            embeddings2 = s_emb2
            total_embeddings = s_emb+s_emb2

            weights_arr = embeddings1.detach().clone()
            weights_arr2 = embeddings2.detach().clone()
            all_embeddings = total_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)
            weights2 = weights_arr2.view(node_num, -1)

            cos_ji_mat = self.relu(torch.matmul(weights, weights2.T))
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights2.norm(dim=-1).view(1,-1)).to(device)
            cos_ji_mat = (cos_ji_mat / normed_mat)

            topk_list_cur = self.topk_list
            topk_indices_ji=[]
            for t in range(self.node_num):
                candidates = cos_ji_mat[t]
                topk = torch.topk(candidates,self.topk_list[t],dim=-1).indices.tolist()
                topk = [i for i in topk if candidates[i]>0]
                topk_list_cur[t] = len(topk)
                topk_indices_ji.append(topk)
            
            gated_i = torch.Tensor([i for i in range(self.node_num) for j in range(topk_list_cur[i])]).unsqueeze(0)
            gated_j = torch.Tensor([i for sublist in topk_indices_ji for i in sublist]).unsqueeze(0)
            
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out1 = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out1)
            

            origin = cos_ji_mat
            cos_ji_mat2 = torch.mul(origin,self.mask1)
            topk_indices_ji2=[]
            topk_list1_cur = self.topk_list1
            for t in range(self.node_num):
                candidates = cos_ji_mat2[t]
                topk = torch.topk(candidates,self.topk_list1[t],dim=-1).indices.tolist()
                topk = [i for i in topk if candidates[i]>0]
                topk_list1_cur[t] = len(topk)
                topk_indices_ji2.append(topk)
                
            gated_i = torch.Tensor([t for t in range(self.node_num) for j in range(topk_list1_cur[t])]).unsqueeze(0)
            gated_j = torch.Tensor([t for sublist in topk_indices_ji2 for t in sublist]).unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out2 = self.gnn_layers[i+1](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings) ##
            
            
            gcn_outs.append(gcn_out2)


            cos_ji_mat3 = torch.mul(origin,self.mask2) ##
            topk_indices_ji3=[]
            topk_list2_cur = self.topk_list2
            for t in range(self.node_num):
                candidates = cos_ji_mat3[t]
                topk = torch.topk(candidates,self.topk_list2[t],dim=-1).indices.tolist()
                topk = [i for i in topk if candidates[i]>0]
                topk_list2_cur[t] = len(topk)
                topk_indices_ji3.append(topk)
                
            gated_i = torch.Tensor([t for t in range(node_num) for j in range(topk_list2_cur[t])]).unsqueeze(0)
            gated_j = torch.Tensor([t for sublist in topk_indices_ji3 for t in sublist]).unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out3 = self.gnn_layers[i+1](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings) ##i+2 아닌가
            
            
            gcn_outs.append(gcn_out3)
        self.cos_g = origin
        self.g = topk_indices_ji
        self.cos_gs = cos_ji_mat2
        self.gs = topk_indices_ji2
        self.cos_gp = cos_ji_mat3
        self.gp = topk_indices_ji3
        x1 = torch.cat([gcn_outs[0]], dim=1)
        x1 = x1.view(batch_num, node_num, -1)
        x2 = torch.cat([gcn_outs[1]], dim=1)
        x2 = x2.view(batch_num, node_num, -1)
        x3 = torch.cat([gcn_outs[2]], dim=1)
        x3 = x3.view(batch_num, node_num, -1)
        out1 = torch.mul(x1, total_embeddings)
        out2 = torch.mul(x2, total_embeddings)
        out3 = torch.mul(x3, total_embeddings)
        out = torch.cat([out1,out2],dim=2)
        out = torch.cat([out,out3],dim=2)
        
    
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out