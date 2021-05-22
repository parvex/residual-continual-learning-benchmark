import unittest
import torch


class TensorMultiplying(unittest.TestCase):
    def test_tensor_multiplying_last_layer(self):
        out = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

        out = out.unsqueeze(0).repeat(128,1,1,1)

        alfa = torch.tensor([2.0, 3.0, 4.0])

        expected = torch.tensor([[[2., 4., 6.],
                                  [8., 10., 12.],
                                  [14., 16., 18.]],

                                 [[3., 6., 9.],
                                  [12., 15., 18.],
                                  [21., 24., 27.]],

                                 [[4., 8., 12.],
                                  [16., 20., 24.],
                                  [28., 32., 36.]]])

        expected = expected.unsqueeze(0).repeat(128, 1, 1, 1)

        result = out * alfa[None, :, None, None]

        self.assertTrue(torch.equal(expected, result))

    def test_tensor_multiplying_prelast_layer(self):
        out = torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
             [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], [[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]], [[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]]])

        alfa = torch.tensor([2.0, 3.0, 4.0])

        expected = torch.tensor([[[2., 4., 6.],
                                  [8., 10., 12.],
                                  [14., 16., 18.]],

                                 [[3., 6., 9.],
                                  [12., 15., 18.],
                                  [21., 24., 27.]],

                                 [[4., 8., 12.],
                                  [16., 20., 24.],
                                  [28., 32., 36.]]])

        result = out * alfa[None, :, None, None]

        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    unittest.main()
