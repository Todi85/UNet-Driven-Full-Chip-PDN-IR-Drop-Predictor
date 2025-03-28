import torch
import torch.nn as nn
import torchvision.ops
import math
from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):

            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)
def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class DSC(nn.Module):
    def __init__(self, in_channels, out_channels,k=3,s=1,p=1,b=True):
        super(DSC, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k, s, p, groups=in_channels,bias=b),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        out = self.main(input)
        return out

class RMC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(RMC, self).__init__()
        self.trunk = DSC(dim_in,dim_out)
        self.branch1 = DSC(dim_in,dim_out,5,1,2)
        self.branch2 = DSC(dim_in,dim_out,7,1,3)
        self.fusion = DSC(2*dim_out,dim_out,3,1,1)
        self.final = DSC(dim_out, dim_out)

    def forward(self, input):
        t = self.trunk(input)
        b1 = self.branch1(input)
        b2 = self.branch2(input)
        b = self.fusion(torch.cat((b1, b2), dim=1))
        out = self.final(t + b)
        return out

class DC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=self.padding,
                                          bias=True)

        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size ** 2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def _init_weight(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.final = DSC(channels, channels)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = self.final(2 * x * wei + 2 * residual * (1 - wei))

        return xo

class US(nn.Module):
    def __init__(self, dim_in, dim_out, k=4, s=2, p=1):
        super(US, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, k, s, p),
            nn.InstanceNorm2d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        out = self.main(input)
        return out

class AG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AG, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.final = DSC(F_g, F_g)

    def forward(self, r, s):
        r1 = self.W_g(r)
        s1 = self.W_x(s)
        psi = self.relu(r1 + s1)
        psi = self.psi(psi)
        out = self.final(s * psi)
        return out

class Rebuilder(nn.Module):
    def __init__(self):
        super(Rebuilder, self).__init__()
        self.layer1 = nn.Sequential(
            DC(4, 8,3,1,1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            DC(1, 8,3,1,1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        '''e1 = input[:, 0:4, :, :]
        e2 = input[:, 4:5, :, :]
        e1 = self.layer1(e1)
        e2 = self.layer2(e2)'''
        e1 = input[:, 0:1, :, :]
        e2 = input[:, 1:5, :, :]
        e1 = self.layer2(e1)
        e2 = self.layer1(e2)
        out = torch.cat([e1,e2],1)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.first = RMC(16, 32)
        self.mdown1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            RMC(32, 64))
        self.mdown2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            RMC(64, 128))
        self.mdown3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            RMC(128, 256))
        self.adown1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            DSC(32, 64))
        self.adown2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            DSC(64, 128))
        self.adown3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            DSC(128, 256))

        self.fd1 = AFF(64, 8)
        self.fd2 = AFF(128, 8)
        self.fd3 = AFF(256, 8)

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        d1 = self.first(input)

        md2 = self.mdown1(d1)
        ad2 = self.adown1(d1)
        d2 = self.fd1(md2,ad2)

        md3 = self.mdown2(d2)
        ad3 = self.adown2(d2)
        d3 = self.fd2(md3,ad3)

        md4 = self.mdown3(d3)
        ad4 = self.adown3(d3)
        d4 = self.fd3(md4,ad4)

        return d1, d2, d3, d4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up3 = nn.Sequential(
            US(256, 128),
            DSC(128, 128))
        self.up2 = nn.Sequential(
            US(256, 64),
            DSC(64, 64))
        self.up1 = nn.Sequential(
            US(128, 32),
            DSC(32, 32))
        self.final = nn.Sequential(
            DSC(64, 16),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid())

        self.s3 = AFF(128, 128)
        self.s2 = AFF(64, 64)
        self.s1 = AFF(32, 32)

        self.ag3 = AG(F_g=128, F_l=128, F_int=64)
        self.ag2 = AG(F_g=64, F_l=64, F_int=32)
        self.ag1 = AG(F_g=32, F_l=32, F_int=16)

        self.u2 = DSC(128, 64)
        self.u1 = DSC(64, 32)
        self.r4s = US(128, 64)
        self.r3s = US(64, 32)

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        d1, d2, d3, d4 = input

        r4 = self.up3(d4)

        a3 = self.ag3(r4, d3)
        s3 = self.s3(a3, d3)
        u3=r4
        r3 = self.up2(torch.cat([u3, s3], dim=1))

        a2 = self.ag2(r3, d2)
        s2 = self.s2(a2, d2)
        u2=self.u2(torch.cat([r3,self.r4s(r4)],1))
        r2 = self.up1(torch.cat([u2, s2], dim=1))

        a1 = self.ag1(r2, d1)
        s1 = self.s1(a1, d1)
        u1=self.u1(torch.cat([r2,self.r3s(r3)],1))
        r1 = self.final(torch.cat([u1, s1], dim=1))

        return r1

class LaRED(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rebuilder = Rebuilder()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.rebuilder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def init_weights(self, pretrained=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
            print('Load state dict form {}'.format(pretrained))
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
