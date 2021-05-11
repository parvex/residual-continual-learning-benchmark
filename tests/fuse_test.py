import unittest
from models.combined import CombinedResNet
import torch


class CombinedFuse(unittest.TestCase):
    def test_combined_fuse(self):
        out = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])

        conv = torch.nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=False)
        conv.weight = torch.nn.Parameter(torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], requires_grad=True))
        bn = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        bn.weight = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        bn.bias = torch.nn.Parameter(torch.tensor([1.5], requires_grad=True))

        expected = bn(conv(out))

        fused_conv = CombinedResNet.fuse(conv, bn)

        result = fused_conv(out)

        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    unittest.main()
