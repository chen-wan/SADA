
import torch
import torch.nn as nn
from einops import rearrange
from torchsummary import summary
import torch.nn.functional as F
from torch.autograd import Function


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return F.relu(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads: int = 9, dropout: float = 0.25):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.projection = nn.Linear(emb_size, emb_size)
        self.att_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads,
                        qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_dropout(att)  # 随机丢弃层
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels * reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TSNet(nn.Module):

    def __init__(self, nChan, nClass, nTime = 1000,
                 dropoutP = 0.25, F1=8, D = 2,
                 C1 = 125, *args, **kwargs):
        super(TSNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        block1_1 = nn.Sequential(
                        nn.Conv2d(1, self.F1, (1, self.C1),
                                  padding=(0, self.C1 // 2), bias=False),
                        nn.BatchNorm2d(self.F1),
                        Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                             padding=0, bias=False, max_norm=1,
                                             groups=self.F1),
                        nn.BatchNorm2d(self.F1 * self.D),
                        nn.ELU(),
                        nn.AvgPool2d((1, 4), stride=4),
                        nn.Dropout(p=dropoutP))

        block1_2 = nn.Sequential(
                        nn.Conv2d(1, self.F1, (1, 63),
                                  padding=(0, 63 // 2), bias=False),
                        nn.BatchNorm2d(self.F1),
                        Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                             padding=0, bias=False, max_norm=1,
                                             groups=self.F1),
                        nn.BatchNorm2d(self.F1 * self.D),
                        nn.ELU(),
                        nn.AvgPool2d((1, 4), stride=4),
                        nn.Dropout(p=dropoutP))

        block1_3 = nn.Sequential(
                        nn.Conv2d(1, self.F1, (1, 31),
                                  padding=(0, 31 // 2), bias=False),
                        nn.BatchNorm2d(self.F1),
                        Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                             padding=0, bias=False, max_norm=1,
                                             groups=self.F1),
                        nn.BatchNorm2d(self.F1 * self.D),
                        nn.ELU(),
                        nn.AvgPool2d((1, 4), stride=4),
                        nn.Dropout(p=dropoutP))

        block2_1 = nn.Sequential(
                        nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 45),
                                  padding=(0, 45 // 2), bias=False,
                                  groups=self.F1 * self.D),
                        nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                                  stride=1, bias=False, padding=0),
                        nn.BatchNorm2d(self.F2),
                        nn.ELU(),
                        nn.AvgPool2d((1, 5), stride=5),
                        nn.Dropout(p=dropoutP)
                        )

        block2_2 = nn.Sequential(
                        nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 21),
                                  padding=(0, 21 // 2), bias=False,
                                  groups=self.F1 * self.D),
                        nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                                  stride=1, bias=False, padding=0),
                        nn.BatchNorm2d(self.F2),
                        nn.ELU(),
                        nn.AvgPool2d((1, 5), stride=5),
                        nn.Dropout(p=dropoutP)
                        )

        block2_3 = nn.Sequential(
                        nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 7),
                                  padding=(0, 7 // 2), bias=False,
                                  groups=self.F1 * self.D),
                        nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                                  stride=1, bias=False, padding=0),
                        nn.BatchNorm2d(self.F2),
                        nn.ELU(),
                        nn.AvgPool2d((1, 5), stride=5),
                        nn.Dropout(p=dropoutP)
                        )

        block3 = nn.Sequential(
                    nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 5),
                              padding=(0, 5 // 2), bias=False,
                              groups=self.F1 * self.D),
                    nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                              stride=1, bias=False, padding=0),
                    nn.BatchNorm2d(self.F2),
                    nn.ELU(),

                    nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 5),
                              padding=(0, 5 // 2), bias=False,
                              groups=self.F1 * self.D),
                    nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                              stride=1, bias=False, padding=0),
                    nn.BatchNorm2d(self.F2),
                    )

        block4 = nn.Sequential(
                    nn.Conv2d(self.F2, self.F1 * self.D, (1, 5),
                              padding=(0, 5 // 2), bias=False,
                              groups=self.F2),
                    nn.Conv2d(self.F2, self.F2, (1, 1),
                              stride=1, bias=False, padding=0),
                    nn.BatchNorm2d(self.F2)
                    )

        block5 = nn.Sequential(
                    nn.Conv2d(self.F2, 20, (1, 10), stride=(1,10)),
                    nn.BatchNorm2d(20),
                    nn.ELU(),
                    )
        block6 = nn.Conv2d(20, 60, (1, 5), stride=(1, 5))

        self.block1_1 = block1_1
        self.block1_2 = block1_2
        self.block1_3 = block1_3
        self.block2_1 = block2_1
        self.block2_2 = block2_2
        self.block2_3 = block2_3
        self.block3 = block3
        self.block4 = block4
        self.block5 = block5
        self.block6 = block6
        self.pool = nn.MaxPool2d((1,5), (1,5))
        self.res_add3 = ResidualAdd(block3)
        self.res_add4 = ResidualAdd(block4)
        self.matt = ChannelAttention(nChan, 2)
        self.pool2 = nn.MaxPool2d((1,2), (1,2))
        self.fc = nn.Linear(60, self.nClass)

    def forward(self, x):
        x = self.block1_1(x)
        # x1 = self.block2_1(x)
        # x1 = self.block5(x1)
        x2 = self.block2_2(x)
        x2 = self.block5(x2)
        # x3 = self.block2_3(x)
        # x3 = self.block5(x3)
        # x = torch.cat([x1, x2, x3], dim=1)
        x = self.block6(x2)
        x = x.view(x.size(0), -1)

        return x


class Classifier(nn.Module):
    def __init__(self,feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x



class Discriminator(nn.Module):
    def __init__(self,feature_dim, num_classes):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim*2),
            nn.ReLU(),
            nn.Linear(in_features=feature_dim*2, out_features=num_classes)
        )

    def forward(self, input_feature):
        x = self.discriminator(input_feature)
        return x



class DiscriminatorF(nn.Module):
    def __init__(self,feature_dim, num_classes):
        super(DiscriminatorF, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim*2),
            nn.ReLU(),
            nn.Linear(in_features=feature_dim*2, out_features=num_classes)  # 输出维度2
        )

    def forward(self, input_feature, alpha):

        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return x



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


if __name__ == '__main__':

    net = TSNet(nChan=3, nClass=2).cuda()
    summary(net,(1, 3, 1000))












