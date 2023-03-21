import torch.nn as nn
import torch.nn.functional as F
from models.utils import ResBlock


class ResNet(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channel = 32

        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(ResBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(ResBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(ResBlock, 256, 2, stride=2)
        self.linear = nn.Linear(256, num_classes)
        self.init_weight()

    def _make_layer(self, block, in_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if in_channel <= 64:
                layers.append(block(self.in_channel, in_channel, stride, norm=False))
            else:
                layers.append(block(self.in_channel, in_channel, stride))
            self.in_channel = in_channel
        return nn.Sequential(*layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, 0)


class EncoderModule(nn.Module):
    def __init__(self, img_channels):
        super(EncoderModule, self).__init__()
        self.layer1 = self._make_layer(img_channels, 32, 3, 1, 1)
        self.layer2 = nn.Sequential(self._make_layer(32, 32, 3, 1, 1))
        self.layer3 = nn.Sequential(self._make_layer(32, 64, 3, 2, 1))

    @staticmethod
    def _make_layer(in_channel, out_channel, k, s, p):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, k, s, p))
        return layer


class DecoderModule(nn.Module):
    def __init__(self, img_channels):
        super(DecoderModule, self).__init__()
        self.layer3 = nn.Sequential(self._make_up_layer(64, 32, 2))
        self.layer2 = nn.Sequential(self._make_layer(32, 32, 3, 1, 1))
        self.layer1 = nn.Sequential(nn.Conv2d(32, img_channels, (3, 3), (1,), 1, bias=False), nn.Tanh())

    @staticmethod
    def _make_up_layer(in_channel, out_channel, up_scale):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(in_channel, out_channel, (1, 1), (1,)))
        return layer

    @staticmethod
    def _make_layer(in_channel, out_channel, k, s, p):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, k, s, p), nn.ReLU(True))
        return layer


class Client(ResNet):
    def __init__(self, img_channels, num_class, level):
        super(Client, self).__init__(img_channels, num_class)
        self.level = level
        # delete inused modules for fit opacus
        if self.level == 1:
            del self.layer1
            del self.layer2
            del self.layer3
            del self.layer4
            del self.linear
        elif self.level == 2:
            del self.layer2
            del self.layer3
            del self.layer4
            del self.linear
        elif self.level == 3:
            del self.layer3
            del self.layer4
            del self.linear

    def forward(self, x):
        if self.level == 1:
            x = F.relu(self.conv1(x))
        elif self.level == 2:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
        elif self.level == 3:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
        return x


class Server(ResNet):

    def __init__(self, img_channels, num_class, level):
        super(Server, self).__init__(img_channels, num_class)
        self.level = level

    def forward(self, x):
        if self.level == 1:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        elif self.level == 2:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        elif self.level == 3:
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        return x


class Encoder(EncoderModule):

    def __init__(self, img_channels, level):
        super(Encoder, self).__init__(img_channels)
        self.level = level

    def forward(self, x):
        if self.level == 1:
            x = self.layer1(x)
            x = nn.ReLU(True)(x)
        elif self.level == 2:
            x = self.layer1(x)
            x = self.layer2(x)
            x = nn.ReLU(True)(x)
        elif self.level == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = nn.ReLU(True)(x)
        return x


class Decoder(DecoderModule):

    def __init__(self, img_channels, level):
        super(Decoder, self).__init__(img_channels)
        self.level = level

    def forward(self, x):
        if self.level == 1:
            x = self.layer1(x)
        elif self.level == 2:
            x = self.layer2(x)
            x = self.layer1(x)
        elif self.level == 3:
            x = self.layer3(x)
            x = self.layer2(x)
            x = self.layer1(x)
        return x


class Gnet(nn.Module):
    def __init__(self, level):
        super(Gnet, self).__init__()
        self.level = level
        if self.level == 1:
            self.model = self.stack_conv2d(num_conv2d=3, fmap_channel=32, sec_chennel=64)
        elif self.level == 2:
            self.model = self.stack_conv2d(num_conv2d=4, fmap_channel=32, sec_chennel=64)
        elif self.level == 3:
            self.model = self.stack_conv2d(num_conv2d=4, fmap_channel=64, sec_chennel=128)

    def forward(self, x):
        return self.model(x)

    def _make_layer(self, in_channel, out_channel, k, s, p):
        layer = [nn.Conv2d(in_channel, out_channel, k, s, p),
                 nn.BatchNorm2d(out_channel),
                 nn.ReLU(True)]
        return layer

    def stack_conv2d(self, num_conv2d, fmap_channel, sec_chennel):
        if num_conv2d == 1:
            return nn.Sequential(*[nn.Conv2d(fmap_channel, fmap_channel, (3, 3), (1,), 1), nn.ReLU(True)])
        layers = self._make_layer(fmap_channel, sec_chennel, 3, 1, 1)
        if num_conv2d - 2 > 0:
            for _ in range(num_conv2d - 2):
                layers += self._make_layer(sec_chennel, sec_chennel, 3, 1, 1)
        layers += [nn.Conv2d(sec_chennel, fmap_channel, (3, 3), (1,), 1), nn.ReLU(True)]
        return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, level):
        super(Discriminator, self).__init__()
        self.level = level
        model = []
        if level == 1:
            model += self._make_layer(32, 128, 3, 2, 1)
            model += self._make_layer(128, 128, 3, 2, 1)
            model += self._make_layer(128, 128, 3, 2, 1)
            model += [nn.Flatten(), nn.Linear(128 * 4 * 4, 1)]
        elif level == 2:
            model += self._make_layer(32, 128, 3, 2, 1)
            model += self._make_layer(128, 128, 3, 2, 1)
            model += self._make_layer(128, 128, 3, 2, 1)
            model += [nn.Flatten(), nn.Linear(128 * 4 * 4, 1)]
        elif level == 3:
            model += self._make_layer(64, 128, 3, 1, 1)
            model += self._make_layer(128, 128, 3, 2, 1)
            model += self._make_layer(128, 128, 3, 2, 1)
            model += [nn.Flatten(), nn.Linear(128 * 4 * 4, 1)]

        self.model = nn.Sequential(*model)
        self.init_weight()

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channel, out_channel, k, s, p):
        layer = [nn.Conv2d(in_channel, out_channel, k, s, p, bias=False),
                 nn.InstanceNorm2d(out_channel),
                 nn.LeakyReLU(0.2, True)]
        return layer

