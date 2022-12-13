import unittest
import os
from openprotein.data.dataset import PTDataFactory
from openprotein.data.process import MaskedConverter
from openprotein.core.config import DataConfig

class PTDataFactoryTest(unittest.TestCase):
    def setUp(self):
        self.args = DataConfig()
        # self.mask = MaskedConverter
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        pass

    def test_PTDataFactory(self):
        df = PTDataFactory(self.args)
        data = df.get_data()
        dataloader = df.get_dataloader()
        for i, j, k in dataloader:
            print(i, j)
            break



if __name__ == "__main__":
    unittest.main()