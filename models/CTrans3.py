
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Gate_attention(nn.Module):
    def __init__(self, dim):
        super(Gate_attention, self).__init__()
        self.psi = nn.Conv3d(dim, 1, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = F.sigmoid(self.psi(F.relu(x)))
        out = out.expand_as(x) * x
        return out
class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patchsize,  in_channels, h, w,d):
        super().__init__()
        # img_size = _pair(img_size)
        # patch_size = _pair(patchsize)
        # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        img_size = [h, w,d ]
        patch_size = _triple(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, h, w,d):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv3d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.h, self.w, self.d = int(h), int(w), int(d)

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w ,d= int(np.sqrt(n_patch)), int(np.sqrt(n_patch)),int(np.sqrt(n_patch))
        # h, w, d = int(np.power(n_patch, 1/3)), int(np.power(n_patch, 1/3)), int(np.power(n_patch, 1/3))
        h, w, d = 8,9,8
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w, d)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    def __init__(self, channel_num):
        super(Attention_org, self).__init__()
        self.vis = True
        self.KV_size = sum(channel_num)
        self.channel_num = channel_num
        self.num_attention_heads = 4

        self.key1 = nn.ModuleList()
        self.key2 = nn.ModuleList()
        self.key3 = nn.ModuleList()
        self.key4 = nn.ModuleList()
        self.value1 = nn.ModuleList()
        self.value2 = nn.ModuleList()
        self.value3 = nn.ModuleList()
        self.value4 = nn.ModuleList()
        self.query = nn.ModuleList()
        # self.value = nn.ModuleList()

        for _ in range(4):
            key1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            key2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            key3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            key4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            value1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            value2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            value3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            value4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            query = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            self.key1.append(copy.deepcopy(key1))
            self.key2.append(copy.deepcopy(key2))
            self.key3.append(copy.deepcopy(key3))
            self.key4.append(copy.deepcopy(key4))
            self.value1.append(copy.deepcopy(value1))
            self.value2.append(copy.deepcopy(value2))
            self.value3.append(copy.deepcopy(value3))
            self.value4.append(copy.deepcopy(value4))
            self.query.append(copy.deepcopy(query))
            # self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(self.KV_size, channel_num[0], bias=False)
        self.out2 = nn.Linear(self.KV_size, channel_num[1], bias=False)
        self.out3 = nn.Linear(self.KV_size, channel_num[2], bias=False)
        self.out4 = nn.Linear(self.KV_size, channel_num[3], bias=False)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)



    def forward(self, emb1,emb2,emb3,emb4, emb_all):
        multi_head_V1_list = []
        multi_head_V2_list = []
        multi_head_V3_list = []
        multi_head_V4_list = []

        multi_head_K1_list = []
        multi_head_K2_list = []
        multi_head_K3_list = []
        multi_head_K4_list = []

        multi_head_Q_list = []
        # multi_head_V_list = []
        if emb1 is not None:
            for (key1, value1) in zip(self.key1, self.value1):
                K1 = key1(emb1)
                multi_head_K1_list.append(K1)
                V1 = value1(emb1)
                multi_head_V1_list.append(V1)
        if emb2 is not None:
            for (key2, value2) in zip(self.key2, self.value2):
                K2 = key2(emb2)
                multi_head_K2_list.append(K2)
                V2 = value2(emb2)
                multi_head_V2_list.append(V2)
        if emb3 is not None:
            for (key3, value3) in zip(self.key3, self.value3):
                K3 = key3(emb3)
                multi_head_K3_list.append(K3)
                V3 = value3(emb3)
                multi_head_V3_list.append(V3)
        if emb4 is not None:
            for (key4, value4) in zip(self.key4, self.value4):
                K4 = key4(emb4)
                multi_head_K4_list.append(K4)
                V4 = value4(emb4)
                multi_head_V4_list.append(V4)
        for query in self.query:
            Q = query(emb_all)
            multi_head_Q_list.append(Q)

        # print(len(multi_head_Q4_list))

        multi_head_K1 = torch.stack(multi_head_K1_list, dim=1) if emb1 is not None else None
        multi_head_K2 = torch.stack(multi_head_K2_list, dim=1) if emb2 is not None else None
        multi_head_K3 = torch.stack(multi_head_K3_list, dim=1) if emb3 is not None else None
        multi_head_K4 = torch.stack(multi_head_K4_list, dim=1) if emb4 is not None else None
        multi_head_V1 = torch.stack(multi_head_V1_list, dim=1) if emb1 is not None else None
        multi_head_V2 = torch.stack(multi_head_V2_list, dim=1) if emb2 is not None else None
        multi_head_V3 = torch.stack(multi_head_V3_list, dim=1) if emb3 is not None else None
        multi_head_V4 = torch.stack(multi_head_V4_list, dim=1) if emb4 is not None else None
        multi_head_Q = torch.stack(multi_head_Q_list, dim=1)

        multi_head_Q = multi_head_Q.transpose(-1, -2) if emb1 is not None else None
        # multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        # multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        # multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q, multi_head_K1) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q, multi_head_K2) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q, multi_head_K3) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q, multi_head_K4) if emb4 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None
        # print(attention_probs4.size())

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else: weights=None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        multi_head_V1 = multi_head_V1.transpose(-1, -2)
        multi_head_V2 = multi_head_V2.transpose(-1, -2)
        multi_head_V3 = multi_head_V3.transpose(-1, -2)
        multi_head_V4 = multi_head_V4.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V1) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V2) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V3) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V4) if emb4 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        return O1,O2,O3,O4, weights




class Mlp(nn.Module):
    def __init__(self,in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self,  channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = 4
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        # self.attn_norm =  LayerNorm(config.KV_size,eps=1e-6)
        self.attn_norm = LayerNorm(sum(channel_num), eps=1e-6)
        self.channel_attn = Attention_org( channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.ffn1 = Mlp(channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(channel_num[2],channel_num[2]*expand_ratio)
        self.ffn4 = Mlp(channel_num[3],channel_num[3]*expand_ratio)


    def forward(self, emb1,emb2,emb3,emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1,cx2,cx3,cx4, weights = self.channel_attn(cx1,cx2,cx3,cx4,emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self,channel_num):
        super(Encoder, self).__init__()
        self.vis = True
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        for _ in range(4):
            layer = Block_ViT( channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3,emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1,emb2,emb3,emb4, weights = layer_block(emb1,emb2,emb3,emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1,emb2,emb3,emb4, attn_weights


class ChannelTransformer2_Qcat(nn.Module):
    def __init__(self, img_size=[8,9,8], skip_indices=[0,1,2,3], channel_num=[64, 128, 256, 512], patchSize=[16 , 8, 4, 2]): #,patchSize=[32, 16, 8, 4]):
        super().__init__()
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        h = img_size[0]
        w = img_size[1]
        d = img_size[2]
        # self.LK_attention_1 = AttentionModule(channel_num[0])
        # self.LK_attention_2 = AttentionModule(channel_num[1])
        # self.LK_attention_3 = AttentionModule(channel_num[2])
        # self.LK_attention_4 = AttentionModule(channel_num[3])
        self.LK_attention_1 = Gate_attention(channel_num[0])
        self.LK_attention_2 = Gate_attention(channel_num[1])
        self.LK_attention_3 = Gate_attention(channel_num[2])
        self.LK_attention_4 = Gate_attention(channel_num[3])

        # self.LK_attention_1 = AttentionModule_split4(channel_num[0])
        # self.LK_attention_2 = AttentionModule_split4(channel_num[1])
        # self.LK_attention_3 = AttentionModule_split4(channel_num[2])
        # self.LK_attention_4 = AttentionModule_split4(channel_num[3])


        # self.embeddings_1 = Channel_Embeddings(self.patchSize_1, img_size=img_size,    in_channels=channel_num[0])
        # self.embeddings_2 = Channel_Embeddings(self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1])
        # self.embeddings_3 = Channel_Embeddings(self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2])
        # self.embeddings_4 = Channel_Embeddings(self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3])
        if skip_indices == [0, 1, 7, 8]:
            self.embeddings_1 = Channel_Embeddings(self.patchSize_1, in_channels=channel_num[0] , h=h, w=w, d=d)
            self.embeddings_2 = Channel_Embeddings(self.patchSize_2, in_channels=channel_num[1],  h=h//2, w=w//2, d= d//2)
            self.embeddings_3 = Channel_Embeddings(self.patchSize_3, in_channels=channel_num[2] ,h=h//2, w=w//2, d= d//2)
            self.embeddings_4 = Channel_Embeddings(self.patchSize_4, in_channels=channel_num[3], h=h, w=w, d=d)
            self.encoder = Encoder(channel_num)

            self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                             scale_factor=(self.patchSize_1, self.patchSize_1, self.patchSize_1),
                                             h=h//4 , w=w//4 , d=d//4 )
            self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                             scale_factor=(self.patchSize_2, self.patchSize_2, self.patchSize_2),
                                             h=h//4 , w=w//4 , d=d//4 )
            self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                             scale_factor=(self.patchSize_3, self.patchSize_3, self.patchSize_3),
                                             h=h//4 , w=w//4 , d=d//4 )
            self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
                                             scale_factor=(self.patchSize_4, self.patchSize_4, self.patchSize_4),
                                             h=h//4 , w=w//4 , d=d//4 )
        else:
            self.embeddings_1 = Channel_Embeddings(self.patchSize_1, in_channels=channel_num[0], h=h, w=w, d=d)
            self.embeddings_2 = Channel_Embeddings(self.patchSize_2, in_channels=channel_num[1], h=h//2, w=w//2, d= d//2)
            self.embeddings_3 = Channel_Embeddings(self.patchSize_3, in_channels=channel_num[2], h=h, w=w, d=d)
            self.embeddings_4 = Channel_Embeddings(self.patchSize_4, in_channels=channel_num[3],h=h*2, w=w*2, d=d*2)
            self.encoder = Encoder(channel_num)

            self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1,self.patchSize_1),
                                                 h=h/2, w= w/2, d= d/2)
            self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2, self.patchSize_2),
                                                 h=h/2, w= w/2, d= d/2)
            self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3, self.patchSize_3),
                                                 h=h/2, w= w/2, d= d/2)
            self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,scale_factor=(self.patchSize_4,self.patchSize_4, self.patchSize_4),
                                                 h=h/2, w= w/2, d= d/2)

    def forward(self,en1,en2,en3,en4):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)
        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1,emb2,emb3,emb4)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        en1 = self.LK_attention_1(en1)
        en2 = self.LK_attention_2(en2)
        en3 = self.LK_attention_3(en3)
        en4 = self.LK_attention_4(en4)
        x1 = x1 + en1  if en1 is not None else None
        x2 = x2 + en2  if en2 is not None else None
        x3 = x3 + en3  if en3 is not None else None
        x4 = x4 + en4  if en4 is not None else None
        # return en1, en2, en3, en4

        return x1, x2, x3, x4, attn_weights
if __name__ == '__main__':
    # x1 = torch.rand(1,128,16,18,16)
    # x2 = torch.rand(1,256,8,9,8)
    # x3 = torch.rand(1,64,16,18,16)
    # x4 = torch.rand(1,32,32,36,32)

    # x1 = torch.rand(1,64,32,36,32).to('cuda:0')
    # x2 = torch.rand(1,128,16,18,16).to('cuda:0')
    # x3 = torch.rand(1,32,32,36,32).to('cuda:0')
    # x4 = torch.rand(1,16,64,72,64).to('cuda:0')

    x1 = torch.rand(1,32,64,72,64).to('cuda:0')
    x2 = torch.rand(1,64,32,36,32).to('cuda:0')
    x3 = torch.rand(1,16,64,72,64).to('cuda:0')
    x4 = torch.rand(1,8,128,144,128).to('cuda:0')

    # x1 = torch.rand(1,16,128,144,128).to('cuda:0')
    # x2 = torch.rand(1,32,64,72,64).to('cuda:0')
    # x3 = torch.rand(1,16,64,72,64).to('cuda:0')
    # x4 = torch.rand(1,8,128,144,128).to('cuda:0')
    transformer = ChannelTransformer2_Qcat(img_size=[64,72,64],
                                          channel_num=[32,64,16,8], patchSize=[8, 4, 8, 16],#patchSize=[2, 1, 2, 4],
                                          skip_indices=[3,4,5,6]).to('cuda:0')
    # transformer = ChannelTransformer2_Qcat(img_size=[128,144,128],
    #                                       channel_num=[16,32,16,8], patchSize=[16, 8, 8, 16],
    #                                       skip_indices=[0,1,7,8]).to('cuda:0')
    out1, out2, out3, out4,atten  = transformer(x1, x2, x3, x4)
    print(out1.size())
    print(out2.size())
    print(out3.size())
    print(out4.size())