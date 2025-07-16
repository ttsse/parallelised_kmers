import os
from data_modelling import results_gen
import warnings
import unittest

def disable_DeprecationWarning(fn):
    def _wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fn(*args, **kwargs)
    return _wrapped

class TestData(unittest.TestCase):
    @disable_DeprecationWarning
    def test_fake_data(self):     # test for spurious input for file naming convention
        case = results_gen('all', '22', 'absence')
        with self.assertRaises(FileNotFoundError):
            case.dataImport()
    def test_real_data(self): # test for valid file naming input
        case = results_gen('all', '17', 'absence')
        self.assertIsNotNone(case.dataImport())
    def test_model_no_error(self):  # test for successful model training given valid file
        case = results_gen('all', '17', 'absence')
        self.assertIsNotNone(case.modelTraining())
    def test_model_error(self): # test for error raised in model training given spurious file name input
        case = results_gen('all', '22', 'absence')
        with self.assertRaises(FileNotFoundError):
            case.modelTraining()
    def test_model_output_produced(self): # test for accurate model output produced
        k = '17'
        predicting = 'absence'
        kmer = 'all'
        method = 'svm'
        cross_validation_results_file_path = os.getcwd() + '/models/hyperparams/' + 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        classification_results_file_path = os.getcwd() + '/models/classification_report/' + 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        confusion_matrix_file_path = os.getcwd() + '/models/confusion_matrix/'+ 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.png'
        model_file_path = os.getcwd()+'/models/k'+k+'_'+kmer+'_'+predicting+'_'+method+'.joblib'
        case = results_gen(kmer, k, predicting)
        case.modelTesting()
        self.assertTrue(os.path.isfile(cross_validation_results_file_path))
        self.assertTrue(os.path.isfile(classification_results_file_path))
        self.assertTrue(os.path.isfile(confusion_matrix_file_path))
        self.assertTrue(os.path.isfile(model_file_path))
    def test_performance_recording(self): # test for accurate performance metric recording: valid values are non-zero and positive
        case = results_gen('all', '17', 'absence')
        case.modelTraining()
        training_time, max_training_memory = case.performanceMetrics()
        self.assertTrue(training_time>0)
        self.assertTrue(max_training_memory>0)

unittest.main()