import unittest
import os

from openprotein.piplines.train import Train

class UnirefTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.train = Train(None, None, None, None)

    def test_fit(self):
        self.train.fit(reduction="mean", ingore_index=None)


if __name__ == "__main__":
    unittest.main()