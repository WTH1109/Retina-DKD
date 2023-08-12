from torch import nn


# input b * 1 * 1024 * 1024
# output b * 128 * 1 * 1
class SegNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SegNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # 1 * 1024 * 1024
        self.seq1 = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 16 * 256 * 256
        self.seq2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 32 * 64 * 64
        self.seq3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 64 * 16 * 16
        self.seq4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 128 * 4 * 4

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        return x


class SegFeature(nn.Module):

    def __init__(self, in_channel):
        super(SegFeature, self).__init__()
        self.in_channel = in_channel
        self.seq1 = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 16 * 512 * 512
        self.seq2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 32 * 256 * 256
        self.seq3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 64 * 256 * 256

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        return x


class FeatureMatch(nn.Module):

    def __init__(self, in_channel, out_channel, downsample):
        super(FeatureMatch, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.d_stride = downsample
        self.seq1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=self.d_stride, padding=1)
        )

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        return x
