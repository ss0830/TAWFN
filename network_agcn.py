import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pool import MHA
from torch_geometric.nn import global_max_pool as gmp
from torch.nn import Parameter

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 确保输入维度可以被头数整除
        assert input_dim % num_heads == 0

        # 线性变换层
        self.query_linear = nn.Linear(input_dim, num_heads * head_dim)
        self.key_linear = nn.Linear(input_dim, num_heads * head_dim)
        self.value_linear = nn.Linear(input_dim, num_heads * head_dim)

        # 最后的线性变换层
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 线性变换
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)

        # 将输出 reshape 成多个头
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 加权求和
        attention_output = torch.matmul(attention_weights, values)

        # 将多个头的结果拼接在一起
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                              self.num_heads * self.head_dim)

        # 最后的线性变换
        output = self.output_linear(attention_output)

        return output

class ProteinBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ProteinBiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Bi-LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        # Concatenate the final hidden states from both directions
        concatenated = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)

        # Fully connected layer
        output = self.fc(concatenated)

        return output
class SoftWeighted(nn.Module):
    def __init__(self, num_view):
        super(SoftWeighted, self).__init__()
        self.num_view = num_view
        self.weight_var = Parameter(torch.ones(num_view))
    def forward(self, data):
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in range(self.num_view)]
        # weight_var = 0.50
        high_level_preds = torch.zeros_like(data[0])  # Assuming data[i] are of the same shape
        for i in range(self.num_view):
            high_level_preds += weight_var[i] * data[i]

        return high_level_preds, weight_var

class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[512,512,512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]
        # gcn_layers = []
        # for i in range(1, len(gcn_dims)):
        #     # Define two graph convolutional layers for each desired scale
        #     gcn_layers.append(GCNConv(gcn_dims[i - 1], gcn_dims[i], bias=True))
        #     gcn_layers.append(GCNConv(gcn_dims[i - 1], gcn_dims[i], bias=True))  # Add a second layer for another scale

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling

        if self.pooling == "MHA":
            self.pool = MHA(512, 256, 512, None, 10000, 0.25, ['GMPool_G', 'GMPool_G'], num_heads=8, layer_norm=True)

        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)
    

    def forward(self, x ,data, pertubed=False):
        #x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)


        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))

            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long()))


            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1

        if self.pooling == 'MHA':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())

        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x


        return n_level_feat, g_level_feat


class AGCN(torch.nn.Module):
    def __init__(self, out_dim, weight1,esm_embed=True, pooling='MHA', pertub=False):
        super(AGCN,self).__init__()
        self.esm_embed = esm_embed
        self.pertub = pertub
        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)
        self.pooling = pooling
        self.weight1 = weight1
        # self.multihead_attention = SelfAttention(512, 256)

        # self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        # self.msma = MSMA(input_dim=512, hidden_dim=512, num_classes=out_dim, num_head=4)
        # self.lstm_model = ProteinBiLSTM(512, 16, 2, 512)
        #self.proj_spot = nn.Linear(19, 512)
        if esm_embed:
            self.proj_esm = nn.Linear(1280, 512)
            self.gcn = GraphCNN(pooling=pooling)
        else:
            self.gcn = GraphCNN(pooling=pooling)
        #self.esm_g_proj = nn.Linear(1280, 512)
        self.readout = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, out_dim),
                        nn.Sigmoid()
        )
        
        self.softmax = nn.Softmax(dim=-1)
     
    def forward(self, data):

        x_aa = self.one_hot_embed(data.native_x.long())
        x_aa = self.proj_aa(x_aa)
        if self.esm_embed:
            x = data.x.float()
            x_esm = self.proj_esm(x)
            # x_ls = x_esm.unsqueeze(0)
            # x_ls = self.multihead_attention(x_ls)
            # x_ls = x_ls.permute(1, 0, 2)
            # x_ls = self.lstm_model(x_ls)
            # x_ls = x_ls.squeeze(1)
            x = F.relu(x_aa + x_esm)
            x_esm = F.relu(x_esm)

        else:
            x = F.relu(x_aa)

        gcn_n_feat1, gcn_g_feat1 = self.gcn(x, data)
        gcn_n_feat3, gcn_g_feat3 = self.gcn(x_esm, data)

        soft_weighted = SoftWeighted(num_view=2)

        if self.pertub:
            gcn_n_feat2, gcn_g_feat2 = self.gcn(x, data, pertubed=True)
            # gcn_n_feat2, gcn_g_feat2 = self.gcn(x, data, pertubed=True)
            gcn_n_feat4, gcn_g_feat4 = self.gcn(x_esm, data, pertubed=True)

            gcn_g_feat1, wg1 = soft_weighted([gcn_g_feat1, gcn_g_feat3])
            gcn_g_feat2, _ = soft_weighted([gcn_g_feat2, gcn_g_feat4])

            y_pred = self.readout(gcn_g_feat1)
            return y_pred,  gcn_g_feat1, gcn_g_feat2, wg1
            # return y_pred, gcn_g_feat1,gcn_g_feat2
        else:
            # gcn_g_feat1, _= soft_weighted([gcn_g_feat1, gcn_g_feat3])
            gcn_g_feat1 = self.weight1[0] * gcn_g_feat1 + self.weight1[1] * gcn_g_feat3
            y_pred = self.readout(gcn_g_feat1)
            # y_pred, _ = soft_weighted([y_pred1, y_pred_msma])
            return y_pred
