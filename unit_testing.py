import unittest
from data_modelling import results_gen

class TestDataImport(unittest.TestCase):
    def test_fake_data(self):
        with self.assertRaises(FileNotFoundError):
            results_gen('all', '22', 'absence')
    def test_real_data(self):
        self.assertIsNone(results_gen('all', '17', 'absence'))
            
unittest.main()
