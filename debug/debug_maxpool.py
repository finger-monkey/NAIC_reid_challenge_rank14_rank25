from torch import nn
import torch


def main():
    pool1 = nn.AdaptiveMaxPool2d(output_size=(2, 1))
    pool2 = nn.MaxPool2d(kernel_size=(128, 128))

    input = torch.randn(1, 64, 128, 128)
    input2 = torch.randn(1, 64, 32, 32)
    o1 = pool1(input)
    o2 = pool2(input)
    o3 = pool1(input2)

    print(o1.shape)
    print(o2.shape)
    print(o3.shape)


if __name__ == '__main__':
    main()
