import unittest

from openprotein.utils.errno import FileNotFoundError

class ErrnoTest(unittest.TestCase):

    def test_FileNotFoundError(self):
        try:
            raise FileNotFoundError("./das")
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    unittest.main()