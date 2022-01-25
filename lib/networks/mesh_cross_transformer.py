import torch.nn as nn
import torch.nn.functional as F
import torch

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                 SparseConvTranspose2d,
                                 SparseConvTranspose3d, SparseInverseConv2d,
                                 SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

from lib.config import cfg
from lib.networks.encoder import SpatialEncoder
import pdb
import gc
import math

class SpatialKeyValue(nn.Module):

    def __init__(self):
        super(SpatialKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(256, 256, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x))


def combine_interleaved(t, num_input=4, agg_type="average"):

    t = t.reshape(-1, num_input, *t.shape[1:])

    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.encoder = SpatialEncoder()

        if cfg.weight == 'cross_transformer':
            self.spatial_key_value_0 = SpatialKeyValue()
            self.spatial_key_value_1 = SpatialKeyValue()

        self.xyzc_net = SparseConvNet()
        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(384, 256, 1)

        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)

        self.view_fc = nn.Conv1d(283, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

        self.fc_3 = nn.Conv1d(256, 256, 1)
        self.fc_4 = nn.Conv1d(128, 128, 1)

        self.alpha_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)

        self.rgb_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)
        self.rgb_res_1 = nn.Conv1d(cfg.img_feat_size, 128, 1)

    def cross_attention(self, holder, pixel_feat):

        key_embed, value_embed = self.spatial_key_value_0(
            pixel_feat.permute(2, 1, 0))

        query_key, query_value = self.spatial_key_value_1(
            holder.permute(2, 1, 0))
        k_emb = key_embed.size(1)
        A = torch.bmm(key_embed.transpose(1, 2), query_key)
        A = A / math.sqrt(k_emb)
        A = F.softmax(A, dim=1)
        out = torch.bmm(value_embed, A)

        final_holder = query_value.permute(2, 1, 0) + out.permute(2, 1, 0)

        return final_holder

    def forward(self, pixel_feat, sp_input, grid_coords, holder=None):

        feature = sp_input['feature']
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        p_features = grid_coords.transpose(1, 2)
        grid_coords = grid_coords[:, None, None]

        xyz = feature[..., :3]

        B = cfg.test.batch_size
        n_input = int(pixel_feat.shape[0] / B)

        xyzc_features_list = []

        for view in range(n_input):
            xyzc = SparseConvTensor(holder[view], coord, out_sh, batch_size)
            xyzc_feature = self.xyzc_net(xyzc, grid_coords)
            xyzc_features_list.append(xyzc_feature)

        xyzc_features = torch.cat(xyzc_features_list, dim=0)

        net = self.actvn(self.fc_0(xyzc_features))
        net = self.cross_attention(net,
                                   self.actvn(self.alpha_res_0(pixel_feat)))

        net = self.actvn(self.fc_1(net))

        inter_net = self.actvn(self.fc_2(net))

        opa_net = combine_interleaved(
            inter_net, n_input, "average"
        )
        opa_net = self.actvn(self.fc_3(opa_net))
        alpha = self.alpha_fc(opa_net)
        alpha = alpha.transpose(1, 2)

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()

        return alpha


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(64, 64, 'subm0')
        self.down0 = stride_conv(64, 64, 'down0')

        self.conv1 = double_conv(64, 64, 'subm1')
        self.down1 = stride_conv(64, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x, grid_coords):

        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()
        feature_4 = F.grid_sample(net4,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        '''

        '''

        features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                             dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     3,
                     2,
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
