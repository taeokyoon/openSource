import unittest
from calculator import add, multiply

class TestCalculator(unittest.TestCase):
    def test_add(self):
        # Test addition function
        self.assertEqual(add(3, 3), 6)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
        print("Test add passed")

if __name__ == "__main__":
    unittest.main()
