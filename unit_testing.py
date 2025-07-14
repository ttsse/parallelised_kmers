import unittest
import os
from data_modelling import results_gen

class TestDataImport(unittest.TestCase):
    def test_fake_data(self):
        case = results_gen('all', '22', 'absence')
        with self.assertRaises(FileNotFoundError):
            case.dataImport()
    def test_real_data(self):
        case = results_gen('all', '17', 'absence')
        self.assertIsNotNone(case.dataImport())
    def test_model_output_produced(self):
        k = '17'
        predicting = 'absence'
        kmer = 'all'
        method = 'l2'
        file_path = os.getcwd() + '/models/confusion_matrix/'+ 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.png'
        case = results_gen(kmer, k, predicting)
        case.modelTraining()
        assert os.path.isfile(file_path)
unittest.main()
