import torch
from torch import nn
from .basic import Sequential


class BNInception3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.inception_3a_1x1 = net.inception_3a_1x1
        self.inception_3a_1x1_bn = net.inception_3a_1x1_bn
        self.inception_3a_relu_1x1 = net.inception_3a_relu_1x1
        self.inception_3a_3x3_reduce = net.inception_3a_3x3_reduce
        self.inception_3a_3x3_reduce_bn = net.inception_3a_3x3_reduce_bn
        self.inception_3a_relu_3x3_reduce = net.inception_3a_relu_3x3_reduce
        self.inception_3a_3x3 = net.inception_3a_3x3
        self.inception_3a_3x3_bn = net.inception_3a_3x3_bn
        self.inception_3a_relu_3x3 = net.inception_3a_relu_3x3
        self.inception_3a_double_3x3_reduce = net.inception_3a_double_3x3_reduce
        self.inception_3a_double_3x3_reduce_bn = net.inception_3a_double_3x3_reduce_bn
        self.inception_3a_relu_double_3x3_reduce = net.inception_3a_relu_double_3x3_reduce
        self.inception_3a_double_3x3_1 = net.inception_3a_double_3x3_1
        self.inception_3a_double_3x3_1_bn = net.inception_3a_double_3x3_1_bn
        self.inception_3a_relu_double_3x3_1 = net.inception_3a_relu_double_3x3_1
        self.inception_3a_double_3x3_2 = net.inception_3a_double_3x3_2
        self.inception_3a_double_3x3_2_bn = net.inception_3a_double_3x3_2_bn
        self.inception_3a_relu_double_3x3_2 = net.inception_3a_relu_double_3x3_2
        self.inception_3a_pool = net.inception_3a_pool
        self.inception_3a_pool_proj = net.inception_3a_pool_proj
        self.inception_3a_pool_proj_bn = net.inception_3a_pool_proj_bn
        self.inception_3a_relu_pool_proj = net.inception_3a_relu_pool_proj
        self.inception_3b_1x1 = net.inception_3b_1x1
        self.inception_3b_1x1_bn = net.inception_3b_1x1_bn
        self.inception_3b_relu_1x1 = net.inception_3b_relu_1x1
        self.inception_3b_3x3_reduce = net.inception_3b_3x3_reduce
        self.inception_3b_3x3_reduce_bn = net.inception_3b_3x3_reduce_bn
        self.inception_3b_relu_3x3_reduce = net.inception_3b_relu_3x3_reduce
        self.inception_3b_3x3 = net.inception_3b_3x3
        self.inception_3b_3x3_bn = net.inception_3b_3x3_bn
        self.inception_3b_relu_3x3 = net.inception_3b_relu_3x3
        self.inception_3b_double_3x3_reduce = net.inception_3b_double_3x3_reduce
        self.inception_3b_double_3x3_reduce_bn = net.inception_3b_double_3x3_reduce_bn
        self.inception_3b_relu_double_3x3_reduce = net.inception_3b_relu_double_3x3_reduce
        self.inception_3b_double_3x3_1 = net.inception_3b_double_3x3_1
        self.inception_3b_double_3x3_1_bn = net.inception_3b_double_3x3_1_bn
        self.inception_3b_relu_double_3x3_1 = net.inception_3b_relu_double_3x3_1
        self.inception_3b_double_3x3_2 = net.inception_3b_double_3x3_2
        self.inception_3b_double_3x3_2_bn = net.inception_3b_double_3x3_2_bn
        self.inception_3b_relu_double_3x3_2 = net.inception_3b_relu_double_3x3_2
        self.inception_3b_pool = net.inception_3b_pool
        self.inception_3b_pool_proj = net.inception_3b_pool_proj
        self.inception_3b_pool_proj_bn = net.inception_3b_pool_proj_bn
        self.inception_3b_relu_pool_proj = net.inception_3b_relu_pool_proj
        self.inception_3c_3x3_reduce = net.inception_3c_3x3_reduce
        self.inception_3c_3x3_reduce_bn = net.inception_3c_3x3_reduce_bn
        self.inception_3c_relu_3x3_reduce = net.inception_3c_relu_3x3_reduce
        self.inception_3c_3x3 = net.inception_3c_3x3
        self.inception_3c_3x3_bn = net.inception_3c_3x3_bn
        self.inception_3c_relu_3x3 = net.inception_3c_relu_3x3
        self.inception_3c_double_3x3_reduce = net.inception_3c_double_3x3_reduce
        self.inception_3c_double_3x3_reduce_bn = net.inception_3c_double_3x3_reduce_bn
        self.inception_3c_relu_double_3x3_reduce = net.inception_3c_relu_double_3x3_reduce
        self.inception_3c_double_3x3_1 = net.inception_3c_double_3x3_1
        self.inception_3c_double_3x3_1_bn = net.inception_3c_double_3x3_1_bn
        self.inception_3c_relu_double_3x3_1 = net.inception_3c_relu_double_3x3_1
        self.inception_3c_double_3x3_2 = net.inception_3c_double_3x3_2
        self.inception_3c_double_3x3_2_bn = net.inception_3c_double_3x3_2_bn
        self.inception_3c_relu_double_3x3_2 = net.inception_3c_relu_double_3x3_2
        self.inception_3c_pool = net.inception_3c_pool

    def forward(self, pool2_3x3_s2_out):
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_relu_1x1_out,inception_3a_relu_3x3_out,inception_3a_relu_double_3x3_2_out ,inception_3a_relu_pool_proj_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_relu_1x1_out,inception_3b_relu_3x3_out,inception_3b_relu_double_3x3_2_out,inception_3b_relu_pool_proj_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_relu_3x3_out,inception_3c_relu_double_3x3_2_out,inception_3c_pool_out], 1)
        return inception_3c_output_out


class BNInception4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.inception_4a_1x1 = net.inception_4a_1x1
        self.inception_4a_1x1_bn = net.inception_4a_1x1_bn
        self.inception_4a_relu_1x1 = net.inception_4a_relu_1x1
        self.inception_4a_3x3_reduce = net.inception_4a_3x3_reduce
        self.inception_4a_3x3_reduce_bn = net.inception_4a_3x3_reduce_bn
        self.inception_4a_relu_3x3_reduce = net.inception_4a_relu_3x3_reduce
        self.inception_4a_3x3 = net.inception_4a_3x3
        self.inception_4a_3x3_bn = net.inception_4a_3x3_bn
        self.inception_4a_relu_3x3 = net.inception_4a_relu_3x3
        self.inception_4a_double_3x3_reduce = net.inception_4a_double_3x3_reduce
        self.inception_4a_double_3x3_reduce_bn = net.inception_4a_double_3x3_reduce_bn
        self.inception_4a_relu_double_3x3_reduce = net.inception_4a_relu_double_3x3_reduce
        self.inception_4a_double_3x3_1 = net.inception_4a_double_3x3_1
        self.inception_4a_double_3x3_1_bn = net.inception_4a_double_3x3_1_bn
        self.inception_4a_relu_double_3x3_1 = net.inception_4a_relu_double_3x3_1
        self.inception_4a_double_3x3_2 = net.inception_4a_double_3x3_2
        self.inception_4a_double_3x3_2_bn = net.inception_4a_double_3x3_2_bn
        self.inception_4a_relu_double_3x3_2 = net.inception_4a_relu_double_3x3_2
        self.inception_4a_pool = net.inception_4a_pool
        self.inception_4a_pool_proj = net.inception_4a_pool_proj
        self.inception_4a_pool_proj_bn = net.inception_4a_pool_proj_bn
        self.inception_4a_relu_pool_proj = net.inception_4a_relu_pool_proj
        self.inception_4b_1x1 = net.inception_4b_1x1
        self.inception_4b_1x1_bn = net.inception_4b_1x1_bn
        self.inception_4b_relu_1x1 = net.inception_4b_relu_1x1
        self.inception_4b_3x3_reduce = net.inception_4b_3x3_reduce
        self.inception_4b_3x3_reduce_bn = net.inception_4b_3x3_reduce_bn
        self.inception_4b_relu_3x3_reduce = net.inception_4b_relu_3x3_reduce
        self.inception_4b_3x3 = net.inception_4b_3x3
        self.inception_4b_3x3_bn = net.inception_4b_3x3_bn
        self.inception_4b_relu_3x3 = net.inception_4b_relu_3x3
        self.inception_4b_double_3x3_reduce = net.inception_4b_double_3x3_reduce
        self.inception_4b_double_3x3_reduce_bn = net.inception_4b_double_3x3_reduce_bn
        self.inception_4b_relu_double_3x3_reduce = net.inception_4b_relu_double_3x3_reduce
        self.inception_4b_double_3x3_1 = net.inception_4b_double_3x3_1
        self.inception_4b_double_3x3_1_bn = net.inception_4b_double_3x3_1_bn
        self.inception_4b_relu_double_3x3_1 = net.inception_4b_relu_double_3x3_1
        self.inception_4b_double_3x3_2 = net.inception_4b_double_3x3_2
        self.inception_4b_double_3x3_2_bn = net.inception_4b_double_3x3_2_bn
        self.inception_4b_relu_double_3x3_2 = net.inception_4b_relu_double_3x3_2
        self.inception_4b_pool = net.inception_4b_pool
        self.inception_4b_pool_proj = net.inception_4b_pool_proj
        self.inception_4b_pool_proj_bn = net.inception_4b_pool_proj_bn
        self.inception_4b_relu_pool_proj = net.inception_4b_relu_pool_proj
        self.inception_4c_1x1 = net.inception_4c_1x1
        self.inception_4c_1x1_bn = net.inception_4c_1x1_bn
        self.inception_4c_relu_1x1 = net.inception_4c_relu_1x1
        self.inception_4c_3x3_reduce = net.inception_4c_3x3_reduce
        self.inception_4c_3x3_reduce_bn = net.inception_4c_3x3_reduce_bn
        self.inception_4c_relu_3x3_reduce = net.inception_4c_relu_3x3_reduce
        self.inception_4c_3x3 = net.inception_4c_3x3
        self.inception_4c_3x3_bn = net.inception_4c_3x3_bn
        self.inception_4c_relu_3x3 = net.inception_4c_relu_3x3
        self.inception_4c_double_3x3_reduce = net.inception_4c_double_3x3_reduce
        self.inception_4c_double_3x3_reduce_bn = net.inception_4c_double_3x3_reduce_bn
        self.inception_4c_relu_double_3x3_reduce = net.inception_4c_relu_double_3x3_reduce
        self.inception_4c_double_3x3_1 = net.inception_4c_double_3x3_1
        self.inception_4c_double_3x3_1_bn = net.inception_4c_double_3x3_1_bn
        self.inception_4c_relu_double_3x3_1 = net.inception_4c_relu_double_3x3_1
        self.inception_4c_double_3x3_2 = net.inception_4c_double_3x3_2
        self.inception_4c_double_3x3_2_bn = net.inception_4c_double_3x3_2_bn
        self.inception_4c_relu_double_3x3_2 = net.inception_4c_relu_double_3x3_2
        self.inception_4c_pool = net.inception_4c_pool
        self.inception_4c_pool_proj = net.inception_4c_pool_proj
        self.inception_4c_pool_proj_bn = net.inception_4c_pool_proj_bn
        self.inception_4c_relu_pool_proj = net.inception_4c_relu_pool_proj
        self.inception_4d_1x1 = net.inception_4d_1x1
        self.inception_4d_1x1_bn = net.inception_4d_1x1_bn
        self.inception_4d_relu_1x1 = net.inception_4d_relu_1x1
        self.inception_4d_3x3_reduce = net.inception_4d_3x3_reduce
        self.inception_4d_3x3_reduce_bn = net.inception_4d_3x3_reduce_bn
        self.inception_4d_relu_3x3_reduce = net.inception_4d_relu_3x3_reduce
        self.inception_4d_3x3 = net.inception_4d_3x3
        self.inception_4d_3x3_bn = net.inception_4d_3x3_bn
        self.inception_4d_relu_3x3 = net.inception_4d_relu_3x3
        self.inception_4d_double_3x3_reduce = net.inception_4d_double_3x3_reduce
        self.inception_4d_double_3x3_reduce_bn = net.inception_4d_double_3x3_reduce_bn
        self.inception_4d_relu_double_3x3_reduce = net.inception_4d_relu_double_3x3_reduce
        self.inception_4d_double_3x3_1 = net.inception_4d_double_3x3_1
        self.inception_4d_double_3x3_1_bn = net.inception_4d_double_3x3_1_bn
        self.inception_4d_relu_double_3x3_1 = net.inception_4d_relu_double_3x3_1
        self.inception_4d_double_3x3_2 = net.inception_4d_double_3x3_2
        self.inception_4d_double_3x3_2_bn = net.inception_4d_double_3x3_2_bn
        self.inception_4d_relu_double_3x3_2 = net.inception_4d_relu_double_3x3_2
        self.inception_4d_pool = net.inception_4d_pool
        self.inception_4d_pool_proj = net.inception_4d_pool_proj
        self.inception_4d_pool_proj_bn = net.inception_4d_pool_proj_bn
        self.inception_4d_relu_pool_proj = net.inception_4d_relu_pool_proj
        self.inception_4e_3x3_reduce = net.inception_4e_3x3_reduce
        self.inception_4e_3x3_reduce_bn = net.inception_4e_3x3_reduce_bn
        self.inception_4e_relu_3x3_reduce = net.inception_4e_relu_3x3_reduce
        self.inception_4e_3x3 = net.inception_4e_3x3
        self.inception_4e_3x3_bn = net.inception_4e_3x3_bn
        self.inception_4e_relu_3x3 = net.inception_4e_relu_3x3
        self.inception_4e_double_3x3_reduce = net.inception_4e_double_3x3_reduce
        self.inception_4e_double_3x3_reduce_bn = net.inception_4e_double_3x3_reduce_bn
        self.inception_4e_relu_double_3x3_reduce = net.inception_4e_relu_double_3x3_reduce
        self.inception_4e_double_3x3_1 = net.inception_4e_double_3x3_1
        self.inception_4e_double_3x3_1_bn = net.inception_4e_double_3x3_1_bn
        self.inception_4e_relu_double_3x3_1 = net.inception_4e_relu_double_3x3_1
        self.inception_4e_double_3x3_2 = net.inception_4e_double_3x3_2
        self.inception_4e_double_3x3_2_bn = net.inception_4e_double_3x3_2_bn
        self.inception_4e_relu_double_3x3_2 = net.inception_4e_relu_double_3x3_2
        self.inception_4e_pool = net.inception_4e_pool

    def forward(self, inception_3c_output_out):
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_relu_1x1_out,inception_4a_relu_3x3_out,inception_4a_relu_double_3x3_2_out,inception_4a_relu_pool_proj_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_relu_1x1_out,inception_4b_relu_3x3_out,inception_4b_relu_double_3x3_2_out,inception_4b_relu_pool_proj_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_relu_1x1_out,inception_4c_relu_3x3_out,inception_4c_relu_double_3x3_2_out,inception_4c_relu_pool_proj_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_relu_1x1_out,inception_4d_relu_3x3_out,inception_4d_relu_double_3x3_2_out,inception_4d_relu_pool_proj_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_relu_3x3_out,inception_4e_relu_double_3x3_2_out,inception_4e_pool_out], 1)
        return inception_4e_output_out


class BNInception5(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.inception_5a_1x1 = n.inception_5a_1x1
        self.inception_5a_1x1_bn = n.inception_5a_1x1_bn
        self.inception_5a_relu_1x1 = n.inception_5a_relu_1x1
        self.inception_5a_3x3_reduce = n.inception_5a_3x3_reduce
        self.inception_5a_3x3_reduce_bn = n.inception_5a_3x3_reduce_bn
        self.inception_5a_relu_3x3_reduce = n.inception_5a_relu_3x3_reduce
        self.inception_5a_3x3 = n.inception_5a_3x3
        self.inception_5a_3x3_bn = n.inception_5a_3x3_bn
        self.inception_5a_relu_3x3 = n.inception_5a_relu_3x3
        self.inception_5a_double_3x3_reduce = n.inception_5a_double_3x3_reduce
        self.inception_5a_double_3x3_reduce_bn = n.inception_5a_double_3x3_reduce_bn
        self.inception_5a_relu_double_3x3_reduce = n.inception_5a_relu_double_3x3_reduce
        self.inception_5a_double_3x3_1 = n.inception_5a_double_3x3_1
        self.inception_5a_double_3x3_1_bn = n.inception_5a_double_3x3_1_bn
        self.inception_5a_relu_double_3x3_1 = n.inception_5a_relu_double_3x3_1
        self.inception_5a_double_3x3_2 = n.inception_5a_double_3x3_2
        self.inception_5a_double_3x3_2_bn = n.inception_5a_double_3x3_2_bn
        self.inception_5a_relu_double_3x3_2 = n.inception_5a_relu_double_3x3_2
        self.inception_5a_pool = n.inception_5a_pool
        self.inception_5a_pool_proj = n.inception_5a_pool_proj
        self.inception_5a_pool_proj_bn = n.inception_5a_pool_proj_bn
        self.inception_5a_relu_pool_proj = n.inception_5a_relu_pool_proj
        self.inception_5b_1x1 = n.inception_5b_1x1
        self.inception_5b_1x1_bn = n.inception_5b_1x1_bn
        self.inception_5b_relu_1x1 = n.inception_5b_relu_1x1
        self.inception_5b_3x3_reduce = n.inception_5b_3x3_reduce
        self.inception_5b_3x3_reduce_bn = n.inception_5b_3x3_reduce_bn
        self.inception_5b_relu_3x3_reduce = n.inception_5b_relu_3x3_reduce
        self.inception_5b_3x3 = n.inception_5b_3x3
        self.inception_5b_3x3_bn = n.inception_5b_3x3_bn
        self.inception_5b_relu_3x3 = n.inception_5b_relu_3x3
        self.inception_5b_double_3x3_reduce = n.inception_5b_double_3x3_reduce
        self.inception_5b_double_3x3_reduce_bn = n.inception_5b_double_3x3_reduce_bn
        self.inception_5b_relu_double_3x3_reduce = n.inception_5b_relu_double_3x3_reduce
        self.inception_5b_double_3x3_1 = n.inception_5b_double_3x3_1
        self.inception_5b_double_3x3_1_bn = n.inception_5b_double_3x3_1_bn
        self.inception_5b_relu_double_3x3_1 = n.inception_5b_relu_double_3x3_1
        self.inception_5b_double_3x3_2 = n.inception_5b_double_3x3_2
        self.inception_5b_double_3x3_2_bn = n.inception_5b_double_3x3_2_bn
        self.inception_5b_relu_double_3x3_2 = n.inception_5b_relu_double_3x3_2
        self.inception_5b_pool = n.inception_5b_pool
        self.inception_5b_pool_proj = n.inception_5b_pool_proj
        self.inception_5b_pool_proj_bn = n.inception_5b_pool_proj_bn
        self.inception_5b_relu_pool_proj = n.inception_5b_relu_pool_proj

        self.out_channels = 1024

    def forward(self, inception_4e_output_out):
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_relu_1x1_out,inception_5a_relu_3x3_out,inception_5a_relu_double_3x3_2_out,inception_5a_relu_pool_proj_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_relu_1x1_out,inception_5b_relu_3x3_out,inception_5b_relu_double_3x3_2_out,inception_5b_relu_pool_proj_out], 1)
        return inception_5b_output_out


def bninception(pretrained):
    import pretrainedmodels
    imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
    net = pretrainedmodels.bninception(num_classes=1000, pretrained=imagenet_pretrained)

    layers1 = Sequential(net.conv1_7x7_s2, net.conv1_7x7_s2_bn, net.conv1_relu_7x7, net.pool1_3x3_s2)
    layers2 = Sequential(net.conv2_3x3_reduce, net.conv2_3x3_reduce_bn, net.conv2_relu_3x3_reduce, net.conv2_3x3, net.conv2_3x3_bn, net.conv2_relu_3x3, net.pool2_3x3_s2)
    layers3 = BNInception3(net)
    layers4 = BNInception4(net)
    layers5 = BNInception5(net)

    layers3.out_channels = 576
    layers4.out_channels = 1056
    layers5.out_channels = 1024

    layers = [layers1, layers2, layers3, layers4, layers5]
    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained