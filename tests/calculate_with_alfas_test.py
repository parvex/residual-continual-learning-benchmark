import unittest
from models.combined import CombinedResNet
import torch


class MyTestCase(unittest.TestCase):
    def test_something(self):
        out = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
