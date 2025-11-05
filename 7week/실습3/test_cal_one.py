import unittest
from calculator import add, multiply

class TestCalculator(unittest.TestCase):
    def test_multiply(self):
        # Test multiplication function
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(-1, 5), -5)
        self.assertEqual(multiply(0, 1), 0)
        print("Test multiply passed")

if __name__ == "__main__":
    unittest.main()
