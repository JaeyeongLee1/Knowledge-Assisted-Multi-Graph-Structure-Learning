import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GraphLayer(MessagePassing):
    def __init__(self, in_channel: int, out_channel: int, d_embedding: int,
                 negative_slope=0.2, dropout=0.0, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.w_x = Linear(in_channel, out_channel, bias=False)
        self.w_z = Linear(2*(out_channel + d_embedding), 1, bias=False)
        self.bias = Parameter(torch.empty(out_channel))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.w_x.weight)
        glorot(self.w_z.weight)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        if torch.is_tensor(x):
            x = self.w_x(x)
            x = (x, x)
        else:
            x = (self.w_x(x[0]), self.w_x(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].shape[0])

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights):

        embedding_i, embedding_j = embedding[1][edge_index_i], embedding[0][edges[0]]

        key_i = torch.cat((x_i, embedding_i), dim=-1)
        key_j = torch.cat((x_j, embedding_j), dim=-1)

        key = torch.cat((key_i, key_j), dim=-1)
        pi = self.w_z(key)
        pi = F.leaky_relu(pi, self.negative_slope)

        alpha = softmax(pi, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channel, self.out_channel)
