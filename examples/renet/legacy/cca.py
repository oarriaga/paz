from torch import nn


class CCA(nn.Module):
    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()

        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(
                SepConv4d(
                    in_planes=ch_in,
                    out_planes=ch_out,
                    ksize=k_size,
                    do_padding=True,
                )
            )
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
        # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
        # because of the ReLU layers in between linear layers,
        # this operation is different than convolving a single time with the filters+filters^T
        # and therefore it makes sense to do this.
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(
            0, 1, 4, 5, 2, 3
        )
        return x


class SepConv4d(nn.Module):
    """approximates 3 x 3 x 3 x 3 kernels via two subsequent 3 x 3 x 1 x 1 and 1 x 1 x 3 x 3"""

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=(1, 1, 1),
        ksize=3,
        do_padding=True,
        bias=False,
    ):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)

        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    bias=bias,
                    padding=0,
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_size=(1, ksize, ksize),
                stride=stride,
                bias=bias,
                padding=padding1,
            ),
            nn.BatchNorm3d(in_planes),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_size=(ksize, ksize, 1),
                stride=stride,
                bias=bias,
                padding=padding2,
            ),
            nn.BatchNorm3d(in_planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x
