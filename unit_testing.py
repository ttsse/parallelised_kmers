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
    
    # test for spurious input for file naming convention
    def test_fake_data(self):
        case = results_gen('all', '22', 'absence')
        with self.assertRaises(FileNotFoundError):
            case.dataImport()

    # test for valid file naming input
    def test_real_data(self):
        case = results_gen('all', '17', 'absence')
        self.assertIsNotNone(case.dataImport())

    # test for successful model training given valid file
    def test_model_no_error(self):
        case = results_gen('all', '17', 'absence')
        self.assertIsNotNone(case.modelTraining())

    # test for error raised in model training given spurious file name input
    def test_model_error(self):
        case = results_gen('all', '22', 'absence')
        with self.assertRaises(FileNotFoundError):
            case.modelTraining()

    # test for accurate model output produced
        # divya comment: ok maybe this test doesnt really make sense,,, why would the file be randomly named something else,,,
    def test_model_output_not_produced(self):
        k = '17'
        predicting = 'absence'
        kmer = 'all'
        method = 'svm'
        fake_k = '50'
        cross_validation_results_file_path = os.getcwd() + '/models/hyperparams/' + 'k'+fake_k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        classification_results_file_path = os.getcwd() + '/models/classification_report/' + 'k'+fake_k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        confusion_matrix_file_path = os.getcwd() + '/models/confusion_matrix/'+ 'k'+fake_k+'_'+kmer+'_'+predicting+'_'+method+'.png'
        model_file_path = os.getcwd()+'/models/k'+fake_k+'_'+kmer+'_'+predicting+'_'+method+'.joblib'

        case = results_gen(kmer, k, predicting)
        case.modelTesting()
        self.assertFalse(os.path.isfile(cross_validation_results_file_path))
        self.assertFalse(os.path.isfile(classification_results_file_path))
        self.assertFalse(os.path.isfile(confusion_matrix_file_path))
        self.assertFalse(os.path.isfile(model_file_path))

    # test for accurate model output produced
    def test_model_output_produced(self):
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

    # test for accurate performance metric recording: valid values are non-zero and positive
    def test_performance_recording(self):
        case = results_gen('all', '17', 'absence')
        case.modelTraining()
        training_time, max_training_memory = case.performanceMetrics()
        self.assertTrue(training_time>0)
        self.assertTrue(max_training_memory>0)

unittest.main()