#####################################
 ##  ____  _            _         ##
 ## |  _ \| |          | |        ##
 ## | |_) | | ___   ___| | _____  ##
 ## |  _ <| |/ _ \ / __| |/ / __| ##
 ## | |_) | | (_) | (__|   <\__ \ ##
 ## |____/|_|\___/ \___|_|\_\___/ ##
#####################################
                               
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import attention 

from  torch.nn.utils.parametrizations import spectral_norm as SN

def _upsample(x, size, mode='bilinear'):
    return F.interpolate(x, size, mode=mode)

class NoneLayer(nn.Module):
    """
    Forward and backward grads just pass this layer
    Used in NormLayer, PaddingLayer, ActivationLayer (etc) if x_type is none
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class NormLayer(nn.Module):
    def __init__(self, channels: int, norm_type: str, affine: bool = True, groups: int = 1):
        super().__init__()

        if norm_type == 'none':
            self.norm_layer = NoneLayer()
        elif norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(channels, affine=affine)
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(channels, affine=affine)
        elif norm_type == 'layer':
            self.norm_layer = nn.GroupNorm(num_groups=1, num_channels=channels, affine=affine)
        elif norm_type == 'group':
            self.norm_layer = nn.GroupNorm(num_groups=groups, num_channels=channels, affine=affine)
        else:
            raise NotImplementedError(
                f'norm_type {norm_type} is not implemented')

    def forward(self, x):
        return self.norm_layer(x)

class PaddingLayer(nn.Module):
    def __init__(self, pad_type: str, pad=0):
        super().__init__()

        if pad_type == 'none':
            self.pad_layer = NoneLayer()
        elif pad_type == 'zero':
            self.pad_layer = nn.ZeroPad2d(padding=pad)
        elif pad_type == 'reflection':
            self.pad_layer = nn.ReflectionPad2d(padding=pad)
        elif pad_type == 'replication':
            self.pad_layer = nn.ReplicationPad2d(adding=pad)
        else:
            raise NotImplementedError(
                f'pad_type {pad_type} is not implemented')

    def forward(self, x):
        return self.pad_layer(x)

class ActivationLayer(nn.Module):
    def __init__(self, act_type: str, inplace=True):
        super().__init__()

        if act_type == 'none':
            self.act_layer = NoneLayer()
        elif act_type == 'sigmoid':
            self.act_layer = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act_layer = nn.Tanh()
        elif act_type == 'relu':
            self.act_layer = nn.ReLU(inplace=inplace)
        elif act_type == 'lrelu':
            self.act_layer = nn.LeakyReLU(0.2, inplace=inplace)
        elif act_type == 'rrelu':
            # lower and upper used by default (1/8, 1/3) for simplication
            # and maybe for my lazyness
            self.act_layer = nn.RReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act_layer = nn.GELU()
        elif act_type == 'elu':
            self.act_layer = nn.ELU()
        else:
            raise NotImplementedError(
                f'act_type {act_type} is not implemented')

    def forward(self, x):
        return self.act_layer(x)

class Attention(nn.Module):
    def __init__(self, base_nc, attention_type = 'none'):
        super().__init__()

        if attention_type == 'none':
            self.att = NoneLayer()
        else:
            self.att = attention.get_attention_block(
                base_nc, attention_name=attention_type
            )

    def forward(self, x):
        return self.att(x)

class UpBlock(nn.Module):
    '''
    up_type: upscale, shuffle, transpose
    '''

    def __init__(self, in_nc, out_nc, up_type: str, factor=2, kernel_size=3, act_type:str='gelu',
        norm_type='none', affine=True, norm_groups=1):
        super(UpBlock, self).__init__()

        block = []
        padding = (kernel_size-1)//2
        if up_type == 'upscale':
            block += [nn.UpsamplingBilinear2d(scale_factor=factor)]
            block += [nn.Conv2d(in_nc, out_nc,
                                kernel_size=kernel_size, padding=padding)]
        elif up_type == 'shuffle':
            block += [nn.PixelShuffle(factor)]
            block += [nn.Conv2d(in_nc, out_nc,
                                kernel_size=kernel_size, padding=padding)]
        elif up_type == 'transpose':
            block += [nn.ConvTranspose2d(in_nc, out_nc,
                                         kernel_size=kernel_size, padding=padding)]
        else:
            raise NotImplementedError(f'up_type [{up_type}] is not implemented')

        block += [NormLayer(out_nc, norm_type=norm_type, affine=affine, groups=norm_groups)]
        block += [ActivationLayer(act_type=act_type)]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor):
        return self.block(x)

class BlockCNA(nn.Module):
    def __init__(self,
                in_nc, out_nc, kernel_size, stride=1, groups=1,
                pad_type='zero',
                act_type='none',
                norm_type='none', affine=True, norm_groups=1,
                use_sn=False):
        super().__init__()

        if norm_type in ('none') or norm_type not in ('batch', 'instance'):
            bias = affine
        else:
            bias = False

        self.pad = PaddingLayer(
            pad_type=pad_type,
            pad=(kernel_size-1)//2
        )
        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size=kernel_size,
            stride=stride, padding=0, groups=groups, bias=bias
        )
        self.norm = NormLayer(
            channels=out_nc, norm_type=norm_type,
            affine=affine, groups=norm_groups
        )
        self.act = ActivationLayer(
            act_type=act_type,
            inplace=True
        )

        if use_sn:
            self.conv = SN(self.conv)
            if norm_type != 'none':
                print('Warning! Please do not use sn with norm. i think it not good')

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.act(out)

        return out

    def forward_conv_only(self, x):
        out = self.pad(x)
        out = self.conv(out)

        return out

class ResBlock(nn.Module):
    def __init__(self,
            nc, kernel_size,
            num_multiple: int=1,
            pad_type='zero',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1):
        super().__init__()
        # res types = 'add', 'cat', 'catadd'
        residual = []

        for _ in range(num_multiple-1):
            residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=kernel_size, stride=1,
                pad_type=pad_type,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type)]

        residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=kernel_size, stride=1,
                pad_type=pad_type,
                norm_type='none', affine=affine, norm_groups=norm_groups,
                act_type='none')]
        self.residual = nn.Sequential(*residual)

        self.act = ActivationLayer(act_type=act_type)

    def forward(self, x):
        return self.act(x + self.residual(x))


class ResDWBlock(nn.Module):
    def __init__(self,
        nc, kernel_size, groups=1,
        num_multiple: int=1,
        pad_type='zero',
        act_type='relu',
        norm_type='none', affine=True, norm_groups=1):

        super().__init__()

        residual = []
        
        for _ in range(num_multiple-1):
            residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=1, groups=1,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                act_type=act_type
            )]

            residual += [BlockCNA(
                in_nc=nc, out_nc=nc, kernel_size=kernel_size, groups=groups,
                norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                pad_type=pad_type,
                act_type=act_type
            )]

        residual += [BlockCNA(
            in_nc=nc, out_nc=nc, kernel_size=1,
            norm_type='none', affine=affine, norm_groups=norm_groups,
            act_type='none'
        )]

        self.residual = nn.Sequential(*residual)

        self.act = ActivationLayer(act_type=act_type)

    def forward(self, x):
        return self.act(x + self.residual(x))

class ResTruck(nn.Module):
    def __init__(self,
            nc, kernel_size, groups=1,
            num_multiple: int=1, num_blocks: int=1,
            pad_type='zero',
            act_type='relu',
            norm_type='none', affine=True, norm_groups=1,
            resblock_type='classic'):
        super(ResTruck, self).__init__()
        # res types = 'add', 'cat', 'catadd'
        blocks = []

        if resblock_type == 'classic':
            for _ in range(num_blocks):
                blocks += [ResBlock(
                    nc, kernel_size, pad_type=pad_type,
                    num_multiple=num_multiple,
                    norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                    act_type=act_type
                )]

        elif resblock_type == 'dw':
            for _ in range(num_blocks):
                blocks += [ResDWBlock(
                    nc, kernel_size, groups=groups, pad_type=pad_type,
                    num_multiple=num_multiple,
                    norm_type=norm_type, affine=affine, norm_groups=norm_groups,
                    act_type=act_type
                )]

        self.truck = nn.Sequential(*blocks)

    def forward(self, x):
        return self.truck(x)


##############
## New code ##
##############

class GDWBlock_A(nn.Module):
    def __init__(self,
        base_nc, kernel_size,
        groups,
        pad_type='zero',
        act_type='gelu',
        norm_type='group', affine=True, norm_groups=1,
        attention_type='none',
        multi_res=0.5):

        super().__init__()
        self.multi = multi_res

        self.block1 = BlockCNA(
            base_nc, base_nc, kernel_size=1, groups=1,
            pad_type=pad_type, act_type=act_type,
            norm_type=norm_type, norm_groups=norm_groups, affine=affine
        )

        self.block2 = BlockCNA(
            base_nc, base_nc, kernel_size=kernel_size, groups=groups,
            pad_type=pad_type, act_type=act_type,
            norm_type='none'
        )

        self.att = Attention(base_nc, attention_type=attention_type)

        self.block3 = BlockCNA(
            base_nc, base_nc, kernel_size=1,
            pad_type=pad_type, act_type='none',
            norm_type='none'
        )

        
    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.att(res)
        res = self.block3(res)
        
        return x + self.multi*res

class RBlock_A(nn.Module):
    def __init__(self,
        base_nc, kernel_size,
        groups=1,
        pad_type='zero',
        act_type='gelu',
        norm_type='group', affine=True, norm_groups=1,
        attention_type='none',
        multi_res=0.5):

        super().__init__()
        self.multi = multi_res

        self.block1 = BlockCNA(
            base_nc, base_nc, kernel_size=kernel_size, groups=groups,
            pad_type=pad_type, act_type=act_type,
            norm_type=norm_type, norm_groups=norm_groups, affine=affine
        )

        self.att = Attention(base_nc, attention_type=attention_type)

        self.block2 = BlockCNA(
            base_nc, base_nc, kernel_size=kernel_size, groups=groups,
            pad_type=pad_type, act_type='none',
            norm_type='none'
        )

        
    def forward(self, x):
        res = self.block1(x)
        res = self.att(res)
        res = self.block2(res)
        
        return x + self.multi*res
    

class RBlock_B(nn.Module):
    def __init__(self,
        base_nc, inner_nc,
        kernel_size,
        groups=1,
        pad_type='zero',
        act_type='gelu',
        norm_type='group', affine=True, norm_groups=1,
        multi_res=1.0):

        super().__init__()
        self.multi = multi_res
        if inner_nc is None:
            inner_nc = base_nc

        self.block1 = BlockCNA(
            base_nc, inner_nc, kernel_size=kernel_size, groups=groups,
            pad_type=pad_type, act_type=act_type,
            norm_type=norm_type, norm_groups=norm_groups, affine=affine
        )

        self.block2 = BlockCNA(
            inner_nc, base_nc, kernel_size=kernel_size, groups=groups,
            pad_type=pad_type, act_type='none',
            norm_type='none'
        )

        
    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        
        return x + self.multi*res


class Truck_A(nn.Module):
    def __init__(self, num_blocks, block, block_args:dict):
        super().__init__()

        truck = []
        for _ in range(num_blocks):
            truck += [block(**block_args)]

        self.truck = nn.Sequential(*truck)

    def forward(self, x):
        return self.truck(x)


class UnetBlock(nn.Module):
    def __init__(self, base_nc, m_nc, block, out_nc=None, act_type='gelu', norm_type='none'):
        super().__init__()
        if out_nc is None:
            out_nc = base_nc

        self.down = BlockCNA(
            base_nc, m_nc, kernel_size=3, stride=2, groups=1,
            pad_type='zero',
            act_type=act_type,
            norm_type=norm_type, affine=True, norm_groups=m_nc//4
        )

        self.block = block

        self.upconv = BlockCNA(
            m_nc, m_nc, kernel_size=3, stride=1, groups=1,
            pad_type='zero',
            act_type=act_type,
            norm_type=norm_type, affine=True, norm_groups=m_nc//4
        )

        self.catconv = nn.Conv2d(base_nc + m_nc, out_nc, kernel_size=1)

    def forward(self, x):

        mid = self.down(x)
        mid = self.block(mid)
        mid = _upsample(mid, size=x.shape[2:4])
        mid = self.upconv(mid)

        out = self.catconv(torch.cat((x, mid), axis=1))

        return out # mid + x


class Truck_GDW(nn.Module):
    def __init__(self, num_blocks, base_nc, attention_type='none'):
        super().__init__()

        block = GDWBlock_A
        block_args = {
            'base_nc': base_nc,
            'kernel_size': 3,
            'groups': base_nc//4,
            'act_type': 'gelu',
            'norm_type': 'group',
            'norm_groups': base_nc//4,
            'attention_type': attention_type,
            'multi_res': 4/(num_blocks+5),
        }

        truck = []
        for _ in range(num_blocks):
            truck += [block(**block_args)]

        self.truck = nn.Sequential(*truck)

    def forward(self, x):
        return self.truck(x)

class Truck_R(nn.Module):
    def __init__(self, num_blocks, base_nc, attention_type='none'):
        super().__init__()

        block = RBlock_A
        block_args={
            'base_nc': base_nc,
            'kernel_size': 3,
            'act_type': 'gelu',
            'norm_type': 'group',
            'norm_groups': base_nc//4,
            'attention_type': attention_type,
            'multi_res': 4/(num_blocks+5)
        }

        truck = []
        for _ in range(num_blocks):
            truck += [block(**block_args)]

        self.truck = nn.Sequential(*truck)

    def forward(self, x):
        return self.truck(x)