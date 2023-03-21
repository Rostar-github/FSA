import torch.nn as nn
import functools
import torch


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, s, in_norm=False, norm=True):
        super(ResBlock, self).__init__()
        self.in_norm = in_norm
        self.norm = norm
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), (s, s), 1, bias=False),
            self.norm_layer(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, (3, 3), (1, 1), 1, bias=False),
            self.norm_layer(out_channel)
        )
        if s != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, 1), (s, s), bias=False),
                self.norm_layer(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()

    def norm_layer(self, channel):
        if self.in_norm:
            return nn.InstanceNorm2d(channel)
        elif not self.in_norm and self.norm:
            return nn.BatchNorm2d(channel)
        elif not self.norm:
            return nn.Sequential()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv2d(x)
        return nn.ReLU(True)(x + shortcut)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, num_convblock=0):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.ReLU(True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            )
            if num_convblock != 0:
                upconvblock = self._make_up_conv2d(num_convblock, inner_nc * 2, inner_nc * 2)
                down = [downconv]
                up = [uprelu, upconvblock, upconv, nn.ReLU(True)]
            else:
                down = [downconv]
                up = [uprelu, upconv, nn.ReLU(True)]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1)
            )
            if num_convblock != 0:
                downconvblock = self._make_down_conv2d(num_convblock, input_nc, input_nc)
                upconvblock = self._make_down_conv2d(num_convblock, inner_nc, inner_nc)
                down = [downrelu, downconvblock, downconv]
                up = [uprelu, upconvblock, upconv, upnorm]
            else:
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            )
            if num_convblock:
                downconvblock = self._make_down_conv2d(num_convblock, input_nc, input_nc)
                upconvblock = self._make_down_conv2d(num_convblock, inner_nc * 2, inner_nc * 2)
                down = [downrelu, downconvblock, downconv, downnorm]
                up = [uprelu, upconvblock, upconv, upnorm]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

    def _make_down_conv2d(self, num_conv2d, input_nc, output_nc):
        block = []
        for i in range(num_conv2d):
            block += [nn.Conv2d(input_nc, output_nc, (3, 3), (1,), 1), nn.BatchNorm2d(output_nc), nn.ReLU(True)]
        return nn.Sequential(*block)

    def _make_up_conv2d(self, num_conv2d, input_nc, output_nc):
        block = []
        for i in range(num_conv2d):
            block += [nn.Conv2d(input_nc, output_nc, (3, 3), (1,), 1), nn.BatchNorm2d(output_nc), nn.ReLU(True)]
            input_nc = output_nc
        return nn.Sequential(*block)
