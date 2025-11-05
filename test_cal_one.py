import unittest
from calculator import add, multiply

class TestCalculator(unittest.TestCase):
    def test_add(self):
        # Test addition function
        self.assertEqual(add(3, 4), 7)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
        print("Test add passed")

    def test_multiply(self):
        # Test multiplication function
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(-1, 5), -5)
        self.assertEqual(multiply(0, 1), 0)
        print("Test multiply passed")

    # def test_add_two(self):
    #     # Test addition function
    #     self.assertEqual(add(5, 4), 7)
    #     print("Test add_two passed")

if __name__ == "__main__":
    unittest.main()
