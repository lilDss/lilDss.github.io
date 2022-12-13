import unittest

from openprotein.utils import dtype

class DtypeTest(unittest.TestCase):

    def test_dtype(self):
        x = [["0", "0"], ["1", "2"], "3", "4"]
        result = dtype.convert_to_bytes(x)
        self.assertEqual(result, [[b'0', b'0'], [b'1', b'2'], b'3', b'4'])

        x1 = [[1, "2"], 3, 4]
        result1 = dtype.convert_to_bytes(x1)
        self.assertEqual(result1, [[b'1', b'2'], b'3', b'4'])

        a = dtype.convert_to_str(result)
        self.assertEqual(a, x)
        pass


if __name__ == "__main__":
    unittest.main()