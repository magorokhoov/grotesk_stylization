import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks

class CoolGen_V8B(nn.Module):
    """
    CoolGen_V8
    8 March 2023, Wednesday
    """

    class Truck_R(nn.Module):
        def __init__(self, num_blocks, base_nc, ks=3, att_type='none'):
            super().__init__()

            block = blocks.RBlock_A
            block_args={
                'base_nc': base_nc,
                'kernel_size': ks,
                'act_type': 'gelu',
                'norm_type': 'group',
                'norm_groups': base_nc//4,
                'attention_type': att_type,
                'multi_res': 1.0
            }

            truck = []
            for _ in range(num_blocks):
                truck += [block(**block_args)]

            self.truck = nn.Sequential(*truck)

        def forward(self, x):
            return self.truck(x)
        
    class Truck_GDW(nn.Module):
        def __init__(self, num_blocks, base_nc, att_type='none'):
            super().__init__()

            block = blocks.GDWBlock_A
            block_args = {
                'base_nc': base_nc,
                'kernel_size': 3,
                'groups': base_nc//8,
                'act_type': 'gelu',
                'norm_type': 'group',
                'norm_groups': base_nc//4,
                'attention_type': att_type,
                'multi_res': 1.0
            }

            truck = []
            for _ in range(num_blocks):
                truck += [block(**block_args)]

            self.truck = nn.Sequential(*truck)

        def forward(self, x):
            return self.truck(x)

    class UnetBlock(nn.Module):
        def __init__(self,in_nc, inner_nc, out_nc, num_truck_blocks, att_type, inner_block):
            super().__init__()

            self.convdown = blocks.BlockCNA(
                in_nc, inner_nc, 3, stride=2,
                #groups=2,
                pad_type='zero',
                act_type='gelu',
                norm_type='group', affine=True, norm_groups=inner_nc//4
            )

            self.truck1 = CoolGen_V8B.Truck_GDW(num_blocks=num_truck_blocks, base_nc=inner_nc, att_type=att_type)
            self.inner_block = inner_block
            self.truck2 = CoolGen_V8B.Truck_GDW(num_blocks=num_truck_blocks, base_nc=inner_nc, att_type=att_type)

            self.convcat = blocks.BlockCNA(
                in_nc+inner_nc, out_nc, 3,
                #groups=2,
                pad_type='zero', act_type='gelu',
                norm_type='group', affine=True, norm_groups=inner_nc//4
            )
            # self.convup = blocks.BlockCNA(
            #     inner_nc, inner_nc, 3,
            #     #groups=2,
            #     pad_type='zero',
            #     act_type='gelu',
            #     norm_type='group', affine=True, norm_groups=inner_nc//4
            # )
            # self.convcat = nn.Conv2d(in_nc+inner_nc, out_nc, 1)


        def forward(self, x):
            _, _, h, w = x.shape

            fea = self.convdown(x)
            fea = self.truck1(fea)
            if self.inner_block is not None:
                fea = self.inner_block(fea)
            fea = self.truck2(fea)
            fea = blocks._upsample(fea, (h,w)) 
            # fea = self.convup(fea)

            return self.convcat(torch.cat((x, fea), axis=1))

    def __init__(self, option_arch:dict):
        super().__init__()
        
        in_nc = option_arch['in_nc']
        base_nc = option_arch['base_nc']
        out_nc = option_arch['out_nc']

        num_truck2_blocks = option_arch['num_truck2_blocks']
        num_truck3_blocks = option_arch['num_truck3_blocks']
        num_truck4_blocks = option_arch['num_truck4_blocks']
        num_truck5_blocks = option_arch['num_truck5_blocks']
        att_type = option_arch['attention_type']
        #num_truck6_blocks = option_arch['num_truck6_blocks']

        #attention_type = option_arch['attention_type']

        # 0: 1x down
        self.conv0 = blocks.BlockCNA(
            in_nc, base_nc, 5,
            pad_type='none',
            act_type='gelu',
            #norm_type='group', affine=True, norm_groups=base_nc//4
        )
        # 1: 2x down
        self.conv_down1 = blocks.BlockCNA(
            base_nc, base_nc, 7, stride=2,
            pad_type='none',
            act_type='gelu',
            norm_type='group', affine=True, norm_groups=base_nc//4
        )

        # 2: 4x down
        self.pool = nn.MaxPool2d(2,2, ceil_mode=True)
        self.truck2_down = self.Truck_R(num_blocks=num_truck2_blocks, base_nc=base_nc, att_type=att_type)

        unet_block = None
        unet_block = self.UnetBlock(
            4*base_nc, 4*base_nc, 4*base_nc,
            num_truck5_blocks, att_type=att_type,
            inner_block=unet_block
        )
        unet_block = self.UnetBlock(
            2*base_nc, 4*base_nc, 2*base_nc,
            num_truck4_blocks, att_type=att_type,
            inner_block=unet_block
        )
        self.unet_block = self.UnetBlock(
            base_nc, 2*base_nc, base_nc,
            num_truck3_blocks, att_type=att_type,
            inner_block=unet_block
        )

        self.truck2_up = self.Truck_R(num_blocks=num_truck2_blocks, base_nc=base_nc, att_type=att_type)

        # 1: 2x up
        self.conv_up1 = blocks.BlockCNA(
            base_nc, base_nc, 3,
            pad_type='none',
            act_type='gelu',
            norm_type='group', affine=True, norm_groups=base_nc//4
        )

        # 0: 1x up
        self.conv_up0 = blocks.BlockCNA(
            base_nc, base_nc, 5,
            pad_type='reflection',
            act_type='gelu',
            norm_type='group', affine=True, norm_groups=base_nc//4
        )

        self.conv_hr = blocks.BlockCNA(
            base_nc, base_nc//2, 5,
            pad_type='reflection',
            act_type='gelu',
            #norm_type='group', affine=True, norm_groups=base_nc
        )
        self.last_conv = nn.Conv2d(base_nc//2, out_nc, 3, padding=0)


    def forward(self, x):

        h0, w0 = x.shape[2:4]
        h1, w1 = h0//2, w0//2

        out = self.conv0(x)
        out = self.conv_down1(out)
        out = self.pool(out)

        out = self.truck2_down(out)
        out = self.unet_block(out)
        out = self.truck2_up(out)

        out = blocks._upsample(out, (h1,w1))
        out = self.conv_up1(out)

        out = blocks._upsample(out, (h0+2,w0+2)) 
        out = self.conv_up0(out)

        out = self.conv_hr(out)
        out = self.last_conv(out)

        return out
