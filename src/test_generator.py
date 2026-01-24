import unittest
from generator import TTSSyntheticDataGenerator

class TestGenerator(unittest.TestCase):
    def test_initialization(self):
        keys = ["test_key"]
        gen = TTSSyntheticDataGenerator(keys)
        self.assertEqual(gen.api_keys, keys)
        self.assertEqual(gen.current_key_index, 0)

    def test_no_keys(self):
        with self.assertRaises(ValueError):
            TTSSyntheticDataGenerator([])

if __name__ == "__main__":
    unittest.main()
