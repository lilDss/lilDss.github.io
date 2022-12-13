import unittest
import os

from openprotein.piplines import MetricUnion, Accuracy, MeanSquaredError, Spearman
# from openprotein.piplines.metrics import MeanSquaredError
import torch

class MetricsTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.true1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        self.pred1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.true2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.pred2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        # self.true1 = torch.Tensor(self.true1).cuda()
        # self.pred1 = torch.Tensor(self.pred1).cuda()
        #  说明：torchcpu版本调用cuda会报错，因此测试mse和spm时先注释了上面两行
        #  说明：cuda()解决方法可以参考新增的测试函数
        # self.metrics = Metrics("")
        #  说明：不加operater list会报错

    def test_Accuracy(self):
        self.metrics = Accuracy()
        self.assertTrue(self.metrics(self.true1, self.pred1), 0.0)
        self.assertTrue(self.metrics(self.true2, self.pred2), 0.5)

    def test_Accuracy2(self):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = Accuracy()
        self.true2 = torch.Tensor(self.true2).to(device)
        self.pred2 = torch.Tensor(self.pred2).to(device)
        # Q: 0.0 is not true :0.0 猜测可能是浮点数0的存储问题，因此即使改动 也无法通过测试
        self.assertTrue(self.metrics(self.true2, self.pred2), 0.0)

    def test_MeanSquaredError(self):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        self.metrics = MeanSquaredError()
        self.true1 = [3, -0.5, 2, 7]
        self.pred1 = [2.5, 0.0, 2, 8]
        self.true1 = torch.Tensor(self.true1).to(device)
        self.pred1 = torch.Tensor(self.pred1).to(device)
        self.assertTrue(self.metrics(self.true1, self.pred1), 0.375)


        # self.true2 = [[0.5, 1], [-1, 1], [7, -6]]
        # self.pred2 = [[0, 2], [-1, 2], [8, -5]]
        # self.true2 = torch.Tensor(self.true2).to(device)
        # self.pred2 = torch.Tensor(self.pred2).to(device)
        # print(self.metrics(self.true2, self.pred2))

    def test_Spearman(self):
        # device = ('cuda' if torch.cuda.is_available() else 'cpu')
        # device = "cpu"
        # self.metrics = Spearman()
        self.true1 = [1, 2, 3, 4, 5]
        self.pred1 = [5, 6, 7, 8, 7]
        self.true1 = torch.Tensor(self.true1).to(device)
        self.pred1 = torch.Tensor(self.pred1).to(device)
        self.assertTrue(self.metrics(self.true1, self.pred1), 0.8207826816681233)



    def test_Metrics(self):
        self.assertTrue(self.metrics(self.true1, self.pred1), 0)
        self.assertTrue(self.metrics(self.true2, self.pred2), 0.5)
        print(self.metrics)



if __name__ == "__main__":
    unittest.main()
