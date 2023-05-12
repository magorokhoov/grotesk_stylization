#########################################################################
 ##    _____    __     __                    __   .__                  ##
 ##   /  _  \ _/  |_ _/  |_   ____    ____ _/  |_ |__|  ____    ____   ##
 ##  /  /_\  \\   __\\   __\_/ __ \  /    \\   __\|  | /  _ \  /    \  ##
 ## /    |    \|  |   |  |  \  ___/ |   |  \|  |  |  |(  <_> )|   |  \ ##
 ## \____|__  /|__|   |__|   \___  >|___|  /|__|  |__| \____/ |___|  / ##
 ##         \/                   \/      \/                        \/  ##
#########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_attention_block(dim, attention_name):
    ### Attention ###
    attention_block = None
    if attention_name == 'se':
        attention_block = SELayer(dim, act_layer=nn.LeakyReLU)
    elif attention_name == 'srm':
        attention_block = SRMLayer(dim)
    # elif attention_name == 'scse':
    #     reduction = min(4, max(dim//64, 2))
    #     attention_block = ChannelSpatialSELayer(dim, reduction_ratio=reduction)
    # elif attention_name == 'cbam':
    #     attention_block = CBAM(dim)
    # elif attention_name == 'stam':
    #     attention_block = StAMLayer(dim)
    else:
        raise Exception(f'There is no attention with name {attention_name}')

    return attention_block

class SELayer(nn.Module):
    def __init__(self, channel, act_layer, reduction=16):
        super(SELayer, self).__init__()

        if act_layer == nn.LeakyReLU:
            act = act_layer(0.2, inplace=True)
        else:
            act = act_layer(inplace=True)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        block = (nn.Linear(channel, channel // reduction, bias=True),
                           act, nn.Linear(channel // reduction, channel,
                           bias=True), nn.Sigmoid() )
        self.fc = nn.Sequential(*block)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SRMLayer(nn.Module):
    def __init__(self, channel, act_layer=None, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=True,
                             groups=channel)
        #TODO num_groups: channel//4, channel//2 or just InstanceNorm?
        #self.norm = nn.GroupNorm(num_channels=channel, num_groups=channel//4, affine=True)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        #z = self.norm(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)
