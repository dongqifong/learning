import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_c, kernel_size, r, dummy_x) -> None:
        super().__init__()
        self.in_c = in_c
        self.conv1 = nn.Conv1d(in_channels=self.in_c,
                               out_channels=self.in_c, kernel_size=kernel_size, padding=int(kernel_size//2))
        self.fc1 = nn.Linear(in_features=self.in_c,
                             out_features=int(self.in_c/r))
        self.fc2 = nn.Linear(in_features=int(
            self.in_c/r), out_features=self.in_c)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        dumm_y = self.conv1(dummy_x)
        self.L = dumm_y.shape[-1]

    def forward(self, x_in):
        x_res = self.conv1(x_in)  # (-1,C,L)
        x = self.global_avg_pooling(x_res)  # (-1,C,1)
        x = x.view(-1, self.in_c)  # (-1,C)
        x = self.fc1(x)  # (-1,C/r)
        x = self.relu(x)  # (-1,C/r)
        x = self.fc2(x)  # (-1,C)
        x = self.sigmoid(x)  # (-1,C)
        x = x.view(-1, self.in_c, 1)  # (-1,C,1)
        x = self.scale(x_res, x)  # (-1,C,L)
        x = x_in + x  # (-1,C,L)
        return x  # (-1,C,L)

    def global_avg_pooling(self, x):
        net = nn.AvgPool1d(kernel_size=self.L)
        return net(x)  # (-1,c,1)

    def scale(self, x_res, x):
        return torch.mul(x_res, x)


if __name__ == "__main__":
    in_c = 5
    r = 4
    kernel_size = 7
    dummy_x = torch.randn((2, in_c, 128))

    senet = SEBlock(in_c=in_c, r=r, kernel_size=kernel_size, dummy_x=dummy_x)
    out = senet(dummy_x)
    print(out.shape)
