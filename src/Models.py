'''
Muhammad Monjurul Karim
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.layers import DropPath, PatchEmbed
import math

from src.RSDlayerAttention import Encoder
from src.fft import SpectralGatingNetwork


class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=[0, 0]):
        super(AccidentPredictor, self).__init__()
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0], training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout[1], training=self.training)
        x = self.dense2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_encoding', self.create_pos_encoding(d_model, max_len))

    def create_pos_encoding(self, d_model, max_len):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)  # [1, max_len, d_model]
        return pos_encoding

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pos_encoding[:, :x.size(1)]


class SelfAttAggregate(nn.Module):
    def __init__(self, agg_dim, num_heads=4):
        super(SelfAttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.num_heads = num_heads
        self.pos_encoder = PositionalEncoding(512, 100)
        assert agg_dim % num_heads == 0, "agg_dim must be divisible by num_heads"

        self.depth = agg_dim // num_heads
        self.Wq = nn.Linear(agg_dim, agg_dim, bias=False)
        self.Wk = nn.Linear(agg_dim, agg_dim, bias=False)
        self.Wv = nn.Linear(agg_dim, agg_dim, bias=False)
        self.dense = nn.Linear(agg_dim, agg_dim)

        torch.nn.init.kaiming_normal_(self.Wq.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wk.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wv.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.dense.weight, a=math.sqrt(5))

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        return x

    def forward(self, hiddens):
        # hiddens: [B, agg_dim, T]
        hiddens = hiddens.permute(0, 2, 1)          # [B, T, agg_dim]
        hiddens = self.pos_encoder(hiddens)         # [B, T, agg_dim]
        batch_size = hiddens.size(0)

        query = self.split_heads(self.Wq(hiddens), batch_size)
        key = self.split_heads(self.Wk(hiddens), batch_size)
        value = self.split_heads(self.Wv(hiddens), batch_size)

        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        depth = key.size(-1)
        logits = matmul_qk / math.sqrt(depth)
        weights = F.softmax(logits, dim=-1)

        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.agg_dim)

        output = self.dense(output)

        maxpool = torch.max(output, dim=1)[0]
        avgpool = torch.mean(output, dim=1)
        agg_feature = torch.cat((avgpool, maxpool), dim=1)

        return agg_feature


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0, 0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # 额外 +512 是为了匹配后续拼接（obj_embed + img_embed + img_fft）
        input_dim = input_dim + 512
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        # x: [B, 1, input_dim]
        out, h = self.gru(x, h)
        out = F.dropout(out[:, -1], self.dropout[0], training=self.training)
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1], training=self.training)
        out = self.dense2(out)
        return out, h


class SpatialAttention(torch.nn.Module):

    def __init__(self, h_dim, n_layers):
        super(SpatialAttention, self).__init__()
        self.n_layers = n_layers
        self.q1 = nn.Linear(h_dim, h_dim)
        self.q2 = nn.Linear(h_dim, h_dim)
        self.wk = nn.Linear(h_dim, h_dim)
        self.wv = nn.Linear(h_dim, h_dim)
        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))

    def forward(self, obj_embed, h):
        """
        obj_embed:
            - [B, N_obj, h_dim]
            - 或 [B, h_dim]
            - 或 [B, N_obj, L, h_dim] （比如这里的 [10, 99, 20, 512]）
        h:
            - [n_layers, B, h_dim]
            - 或 [B, h_dim]
        """

        # ---- 规范 obj_embed 形状：最终都变成 [B, N_obj, D] ----
        if obj_embed.dim() == 4:
            # [B, N_obj, L, D] → 在 L 维上做平均 → [B, N_obj, D]
            B, N_obj, L, D = obj_embed.shape
            obj_embed = obj_embed.mean(dim=2)
        elif obj_embed.dim() == 3:
            # [B, N_obj, D]，直接用
            B, N_obj, D = obj_embed.shape
        elif obj_embed.dim() == 2:
            # [B, D] → [B, 1, D]
            obj_embed = obj_embed.unsqueeze(1)
            B, N_obj, D = obj_embed.shape
        else:
            raise RuntimeError(f"SpatialAttention expects obj_embed 2D/3D/4D, got {obj_embed.shape}")

        # ---- 规范 h 形状：拿到两个 query 向量 h0, h1 ----
        if h.dim() == 3:
            # [n_layers, B, D]
            h0 = h[0]                         # 第一层
            if h.size(0) > 1:
                h1 = h[1]                     # 第二层
            else:
                h1 = h0                       # 只有一层就复用
        elif h.dim() == 2:
            # [B, D]，当成只有一层
            h0 = h
            h1 = h
        else:
            raise RuntimeError(f"SpatialAttention expects h 2D or 3D, got {h.shape}")

        # ---- 线性映射成 query、key、value ----
        query1 = self.q1(h0).unsqueeze(1)      # [B, 1, D]
        query2 = self.q2(h1).unsqueeze(1)      # [B, 1, D]
        key = self.wk(obj_embed)               # [B, N_obj, D]
        value = self.wv(obj_embed)             # [B, N_obj, D]

        key_t = key.transpose(1, 2)            # [B, D, N_obj]

        # ---- 这里两个 bmm 的输入都已经明确是 3D ----
        attention_score1 = torch.bmm(query1, key_t) / math.sqrt(D)  # [B, 1, N_obj]
        attention_score2 = torch.bmm(query2, key_t) / math.sqrt(D)  # [B, 1, N_obj]

        attention_scores = self.alpha1 * attention_score1 + self.alpha2 * attention_score2
        attention_weights = F.softmax(attention_scores, dim=-1)     # [B, 1, N_obj]

        # 加权求和得到空间注意力后的 obj 表示
        weighted_obj_embed = torch.bmm(attention_weights, value)    # [B, 1, D]
        return weighted_obj_embed




class CRASH(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True):
        super(CRASH, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa

        # 基础特征编码
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_x3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())

        self.sp_attention = SpatialAttention(self.h_dim, self.n_layers)
        self.rho_1 = torch.nn.Parameter(torch.tensor(1.0))
        self.rho_2 = torch.nn.Parameter(torch.tensor(1.0))

        # GRU 输入: h_dim + h_dim，再在 GRUNet 里 +512
        self.gru_net = GRUNet(h_dim + h_dim, h_dim, 2, n_layers, dropout=[0.5, 0.0])

        if self.with_saa:
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.self_aggregation = SelfAttAggregate(self.h_dim)

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.RSD = Encoder(512, 8)
        # ★ 这里改成 512 维的频域模块，和 img_embed 的维度一致
        self.fftblock = SpectralGatingNetwork(512)

    def _get_rhos(self):
        """
        把可学参数 self.rho_1, self.rho_2 映射成严格 >0 的数，
        避免 log(<=0) 或除以几乎 0 导致 NaN/Inf。
        """
        eps = 1e-6
        rho1 = F.softplus(self.rho_1) + eps   # > 0
        rho2 = F.softplus(self.rho_2) + eps   # > 0
        return rho1, rho2


    def forward(self, x, y, toa, hidden_in=None, nbatch=80, testing=False):
        """
        x:   [B, T, N_obj+1, x_dim]      或 [B, T, N_obj+1, L, x_dim]
        y:   [B, 2]
        toa: [B]
        """
        losses = {'cross_entropy': 0,
                  'total_loss': 0,
                  'log': 0}
        if self.with_saa:
            losses.update({'auxloss': 0})

        all_outputs, all_hidden = [], []

        # ---------- 初始化 hidden ----------
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim))
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)

        h_list = []

        # ---------- 时间维循环 ----------
        for t in range(x.size(1)):
            # 先过一层 MLP 编码
            x_t_raw = self.phi_x(x[:, t])   # 可能是 [B, N+1, 512] 或 [B, N+1, L, 512]

            # ====== 1. 处理图像和物体特征的形状 ======
            if x_t_raw.dim() == 4:
                # [B, N+1, L, D]
                B, Np1, L, D = x_t_raw.shape

                # 图像部分：index 0，形状 [B, L, D] → 对 L 做平均 → [B, 1, D]
                img_seq = x_t_raw[:, 0, :, :]                   # [B, L, D]
                img_embed = img_seq.mean(dim=1, keepdim=True)  # [B, 1, D]

                # 物体部分：剩下 N_obj 个，[B, N_obj, L, D]
                obj_embed_input = x_t_raw[:, 1:, :, :]         # [B, N_obj, L, D]

            elif x_t_raw.dim() == 3:
                # [B, N+1, D]
                B, Np1, D = x_t_raw.shape

                img_embed = x_t_raw[:, 0, :].unsqueeze(1)      # [B, 1, D]
                obj_embed_input = x_t_raw[:, 1:, :]            # [B, N_obj, D]

            else:
                raise RuntimeError(f"Unexpected x_t shape {x_t_raw.shape} in CRASH.forward")

            # ====== 2. 频域模块：对 img_embed 做 FFT gating ======
            # img_embed: [B, 1, D] → [B, D]
            img_tmp = img_embed.squeeze(1)           # [B, D]
            img_fft = self.fftblock(img_tmp)         # [B, D]
            img_fft = img_fft.unsqueeze(1)           # [B, 1, D]
            img_fft = self.phi_x3(img_fft)           # [B, 1, D]

            # ====== 3. 空间注意力：对 obj_embed_input 做注意力 ======
            # obj_embed_input: 2D / 3D / 4D 都交给 SpatialAttention，它内部会自己处理
            obj_embed = self.sp_attention(obj_embed_input, h)  # [B, 1, D]

            # ====== 4. 拼接三个特征，作为 RNN 的输入 ======
            # obj_embed: [B, 1, D]
            # img_embed: [B, 1, D]
            # img_fft : [B, 1, D]
            x_t = torch.cat([obj_embed, img_embed, img_fft], dim=-1)  # [B, 1, 3D]

            h_list.append(h)

            # ====== 5. RSD 模块更新 h ======
            if t == 2:
                h_staked = torch.stack((h_list[t], h_list[t - 1], h_list[t - 2]), dim=0)
                h = self.RSD(h_staked)
            elif t == 3:
                h_staked = torch.stack((h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3]), dim=0)
                h = self.RSD(h_staked)
            elif t == 4:
                h_staked = torch.stack((h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3], h_list[t - 4]), dim=0)
                h = self.RSD(h_staked)
            elif t == 5:
                h_staked = torch.stack(
                    (h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3], h_list[t - 4], h_list[t - 5]), dim=0)
                h = self.RSD(h_staked)
            elif t == 6:
                h_staked = torch.stack(
                    (h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3], h_list[t - 4], h_list[t - 5],
                     h_list[t - 6]), dim=0)
                h = self.RSD(h_staked)
            elif t == 7:
                h_staked = torch.stack(
                    (h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3], h_list[t - 4], h_list[t - 5],
                     h_list[t - 6], h_list[t - 7]), dim=0)
                h = self.RSD(h_staked)
            elif t == 8:
                h_staked = torch.stack(
                    (h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3], h_list[t - 4], h_list[t - 5],
                     h_list[t - 6], h_list[t - 7], h_list[t - 8]), dim=0)
                h = self.RSD(h_staked)
            elif t > 8:
                h_staked = torch.stack(
                    (h_list[t], h_list[t - 1], h_list[t - 2], h_list[t - 3], h_list[t - 4], h_list[t - 5],
                     h_list[t - 6], h_list[t - 7], h_list[t - 8], h_list[t - 9]), dim=0)
                h = self.RSD(h_staked)

            # ====== 6. GRU 预测当前时间步的输出 ======
            output, h = self.gru_net(x_t, h)

            L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
            losses['cross_entropy'] += L3
            all_outputs.append(output)
            all_hidden.append(h[-1])

        # ---------- SAA 辅助分支 ----------
        if self.with_saa:
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1))
            dec = self.predictor_aux(embed_video)
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))

            rho1, rho2 = self._get_rhos()
            L4 = L4 / (rho2 * rho2)
            losses['auxloss'] = L4

        # 重新取一次（也可以复用上面的 rho1, rho2）
        rho1, rho2 = self._get_rhos()
        losses['log'] = torch.log(rho1 * rho2)

        return losses, all_outputs, all_hidden

    def _exp_loss(self, pred, target, time, toa, fps=10.0):
        """
        改进版本：添加数值稳定性检查和梯度裁剪
        """
        device = pred.device

        # ---- 1. 获取标签 ----
        if target.dim() == 2 and target.size(1) == 2:
            l = target[:, 1].float().to(device)
        else:
            cls = target.view(-1).to(device)
            l = (cls == 1).float()

        # ---- 2. 使用 log_softmax 提高数值稳定性 ----
        log_prob = F.log_softmax(pred, dim=1)
        log_prob_pos = log_prob[:, 1]  # log(p_t^b)
        log_prob_neg = log_prob[:, 0]  # log(1 - p_t^b) 的近似

        # ---- 3. 时间衰减权重 ----
        tau = toa.to(device).float()
        delta = (tau - (time + 1)) / fps
        delta_clamped = torch.clamp(delta, min=0.0)
        weight = torch.exp(-0.5 * delta_clamped)

        # ---- 4. 计算 loss（使用 log 空间） ----
        # 正样本: weight * (-log p_t) = -weight * log_prob_pos
        pos_loss = -weight * log_prob_pos
        # 负样本: -log(1 - p_t) ≈ -log_prob_neg (当使用 log_softmax 时)
        neg_loss = -log_prob_neg

        # 混合
        frame_loss_each = l * pos_loss + (1.0 - l) * neg_loss

        # ---- 5. 检查并处理异常值 ----
        if torch.isnan(frame_loss_each).any() or torch.isinf(frame_loss_each).any():
            print(f"Warning: NaN or Inf detected in loss at time {time}")
            print(f"  pred: {pred}")
            print(f"  log_prob: {log_prob}")
            print(f"  weight: {weight}")
            # 用一个安全的默认值替代
            frame_loss_each = torch.where(
                torch.isnan(frame_loss_each) | torch.isinf(frame_loss_each),
                torch.ones_like(frame_loss_each),
                frame_loss_each
            )

        # ---- 6. 平均并缩放 ----
        rho1, rho2 = self._get_rhos()
        loss = frame_loss_each.mean()
        loss = loss / (rho1 * rho1)

        return loss




