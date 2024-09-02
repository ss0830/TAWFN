import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_batch


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

class SeparableConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv, self).__init__()
        self.conv1 =nn.Sequential(nn.Conv1d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias),nn.BatchNorm1d(inplanes))
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x.unsqueeze(2))
        return x.squeeze(2)

class FPN_Module(nn.Module):
    def __init__(self, hidden_dim):
        super(FPN_Module, self).__init__()
        self.sub_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, )
        self.sub_conv2 = nn.Conv1d(hidden_dim , hidden_dim // 2, kernel_size=3, padding=1, )
        self.sub_conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, )
        self.s_conv1 = conv_2(1,1)
        self.s_conv2 = conv_2(1,2)
        self.s_conv3 = conv_2(1,4)
        self.up_conv1 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=1,dilation=1)
        self.up_conv2 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=2,dilation=2)
        self.up_conv3 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=4,dilation=4)
        self.up_conv4 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=8,dilation=8)
        self.out = conv_2(4,1)

    def forward(self, x):
        B,C,N = x.size()
        c1 = self.sub_conv1(x)
        c2 = self.sub_conv2(c1)
        c3 = self.sub_conv3(c2)
        m1 = self.s_conv1(c1.unsqueeze(1)).reshape(B,C,N)
        m2 = self.s_conv2(c2.unsqueeze(1)).reshape(B,C,N)
        m3 = self.s_conv3(c3.unsqueeze(1)).reshape(B,C,N)
        c_out = torch.cat([m1,m2,m3],dim=1)
        d_1 = self.up_conv1(c_out)
        d_2 = self.up_conv2(c_out)
        d_3 = self.up_conv3(c_out)
        d_4 = self.up_conv4(c_out)
        d_out = torch.cat([d_1,d_2,d_3,d_4],dim=1)
        d_out = self.out(d_out.reshape(B,-1,C,N)).reshape(B,C,N)
        return d_out


class MCNN(nn.Module):
    def __init__(self, input_dim,hidden_dim=64, num_classes=256,num_head=4):
        super(MCNN, self).__init__()
        self.num_head = num_head

        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)
        self.proj_esm = nn.Linear(1280, 512)
        # self.lstm_model = ProteinLSTM(512, 16, 2, 512)
        self.emb = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=0),nn.BatchNorm1d(hidden_dim))
        self.ms_f = FPN_Module(hidden_dim)
        self.multi_head = MCAM(self.num_head,hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self,data):

        # x = data.x
        x_aa = self.one_hot_embed(data.native_x.long())
        x_aa = self.proj_aa(x_aa)
        x = data.x.float()
        x_esm = self.proj_esm(x)

        x = F.relu( x_aa + x_esm)

        batch_x, _ = to_dense_batch(x, data.batch)

        x = batch_x.permute(0, 2, 1)
        conv_emb = self.emb(x)

        # multi-scale feats
        conv_ms = self.ms_f(conv_emb)
        conv_x = self.multi_head(conv_emb)
        conv_mha = self.multi_head(conv_ms)
        out = conv_x + conv_mha
        # attn
        # output = self.multi_head(conv_ms)
        output = torch.flatten(out,1)

        output = self.fc_out(output)
        output = torch.sigmoid(output)
        # batch_x = conv_ms.permute(0, 2, 1)
        return output


class MHA(nn.Module):  # multi-head attention

    neigh_k = list(range(3, 21, 2))

    def __init__(self,num_heads,hidden_dim):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.multi_head = nn.ModuleList([
            nn.ModuleList([CPAM(self.neigh_k[i]), SpatialAttention()])
            for i in range(num_heads)
        ])
        # self.high_lateral_attn = nn.Sequential(nn.Linear(num_heads * hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,num_heads))
        self.weight_var = Parameter(torch.ones(num_heads))

    def forward(self, x):
        max_pool = self.maxpool(x)
        avg_pool = self.avgpool(x)
        pool_feats = []
        for id,head in enumerate(self.multi_head):
            weight_cpam = head[0](max_pool,avg_pool)
            self_attn = x * weight_cpam
            weight_sam = head[1](self_attn)
            self_sam = self_attn *weight_sam
            x = x + self_sam

            # output = head(x)
            pool_feats.append(torch.max(x,dim=2)[0])
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in
                      range(self.num_heads)]
        high_pool_fusion = 0
        for i in range(self.num_heads):
            high_pool_fusion += weight_var[i] * pool_feats[i]

        return high_pool_fusion

class CPAM(nn.Module):
    # def __init__(self, k ,pool_types = ['avg','max']):
    def __init__(self, k ,pool_types = ['avg','max']):
        super(CPAM, self).__init__()
        self.pool_types = pool_types
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, max_pool,avg_pool):
        channel_att_sum = 0.
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # channel_att_raw = self.conv(avg_pool.transpose(1, 2)).transpose(1, 2)
                channel_att_raw = self.conv(avg_pool.transpose(1, 2)).transpose(1, 2)
            elif pool_type == 'max':
                channel_att_raw = self.conv(max_pool.transpose(1, 2)).transpose(1, 2)

            channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum)

        return scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        # assert kernel_size in (3,7), "kernel size must be 3 or 7"
        # padding = 3 if kernel_size == 7 else 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(2,1,kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # avg_pool = torch.mean(x, dim=2, keepdim=True)
        # max_pool, _ = torch.max(x, dim=2, keepdim=True)
        max_pool = self.maxpool(x)
        avg_pool = self.avgpool(x)
        conv_x = torch.cat([max_pool, avg_pool], dim=2)
        conv_x = self.conv(conv_x.transpose(1, 2)).transpose(1, 2)
        conv_x = self.sigmoid(conv_x)
        # return self.sigmoid(conv_x)
        return conv_x
def conv_2(in_planes,out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False), nn.BatchNorm2d(out_planes),nn.ReLU(inplace=True))



