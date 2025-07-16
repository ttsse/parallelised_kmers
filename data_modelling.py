import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import time
import tracemalloc


filepath = os.getcwd() + '/data/'

class results_gen():
    def __init__(self, kmer, k, predicting):
        self.kmer = kmer
        self.k = k
        self.predicting = predicting
        self.X_test = None
        self.y_test = None
        self.model = None
        self.method = None
        self.training_time = None
        self.max_memory = None

    # step 1: data import 
        ## unit tests written
    def dataImport(self):
        kmer = self.kmer
        k = self.k
        predicting = self.predicting

        x_train_path = filepath+'10ribo_X_train_readfile_by_'+kmer+'_'+predicting+'_df_k'+k+'.csv'
        x_test_path = filepath+'10ribo_X_test_readfile_by_'+kmer+'_'+predicting+'_df_k'+k+'.csv'
        y_train_path = filepath+'10ribo_y_train_readfile_by_all_kmer_trueribotype_df_k'+k+'.csv'
        y_test_path = filepath+'10ribo_y_test_readfile_by_all_kmer_trueribotype_df_k'+k+'.csv'

        try:
            X_train = pd.read_csv(x_train_path, index_col=0)
            X_test= pd.read_csv(x_test_path, index_col=0)
            y_train = pd.read_csv(y_train_path, index_col=0)
            y_test = pd.read_csv(y_test_path, index_col=0)

            self.y_test = y_test
            self.X_test = X_test
            return X_train, y_train

        except FileNotFoundError as error:
            raise FileNotFoundError(error)
    
    # step 2: model training
        ## svm
    def modelTraining(self):
        X_train, y_train = self.dataImport()
        kmer = self.kmer
        k = self.k
        predicting = self.predicting
        method='svm'
        self.method = method
        
        start_time = time.time()
        tracemalloc.start()
        model = SVC()
        ovo = OneVsOneClassifier(model)

        SVCparam_grid = {
            'estimator__kernel':('sigmoid', 'rbf', 'poly'), 
            'estimator__gamma': [1, 0.1, 0.01, 0.001],
            'estimator__C':[0.1, 1, 10, 100]
        }
        SVCovo_tuned = RandomizedSearchCV(estimator = ovo, param_distributions = SVCparam_grid, refit=True, n_iter = 25, cv=3, verbose=0, random_state=42, n_jobs = -1)

        SVCovo_tuned.fit(X_train, y_train)
        self.model = SVCovo_tuned
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        seconds_spanned_training = end_time - start_time
        self.training_time = seconds_spanned_training
        self.max_memory = peak_mem
        return SVCovo_tuned 

    def modelTesting(self):
        kmer = self.kmer
        k = self.k
        predicting = self.predicting 
        ovo_tuned = self.modelTraining()
        method= self.method
        X_test = self.X_test
        y_test = self.y_test
        
        df = pd.DataFrame(ovo_tuned.cv_results_)
        cross_validation_results_parent_path = os.getcwd() + '/models/hyperparams/'
        if os.path.exists(cross_validation_results_parent_path):
            pass
        else:
            os.makedirs(cross_validation_results_parent_path)
            #print('make cross validation directory')
        cross_validation_results_path = cross_validation_results_parent_path + 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        df.to_csv(cross_validation_results_path, header=True)

        yhat = ovo_tuned.predict(X_test)
        report = classification_report(y_test, yhat, labels=[1, 2, 5, 14, 15, 17, 20, 27, 78, 106], output_dict=True)
        report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
        df = pd.DataFrame(report).transpose()
        classification_results_parent_path = os.getcwd() + '/models/classification_report/'
        if os.path.exists(classification_results_parent_path):
            pass
        else:
            os.makedirs(classification_results_parent_path)
            #print('make classification results directory')
        classification_results_path = classification_results_parent_path + 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        df.to_csv(classification_results_path, header=True)

        cm = confusion_matrix(y_test, yhat, labels=[1, 2, 5, 14, 15, 17, 20, 27, 78, 106])
        cm = ConfusionMatrixDisplay(cm, display_labels = [1, 2, 5, 14, 15, 17, 20, 27, 78, 106])
    
        confusion_matrix_parent_path = os.getcwd() + '/models/confusion_matrix/'
        if os.path.exists(confusion_matrix_parent_path):
            pass
        else:
            os.makedirs(confusion_matrix_parent_path)
            #print('made confusion matrix directory')
        confusion_matrix_path = confusion_matrix_parent_path + 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.png'
        cm.plot()
        plt.savefig(confusion_matrix_path)    

        joblib.dump(ovo_tuned.best_estimator_, os.getcwd()+'/models/k'+k+'_'+kmer+'_'+predicting+'_'+method+'.joblib')

    def performanceMetrics(self):
        kmer = self.kmer
        k = self.k
        predicting = self.predicting
        training_time = self.training_time
        max_training_memory = self.max_memory
        method = self.method

        performance_metrics_path = os.getcwd() + '/models/performance/'
        if os.path.exists(performance_metrics_path):
            pass
        else:
            os.makedirs(performance_metrics_path)
        performance_metrics_results_path = performance_metrics_path + 'k'+k+'_'+kmer+'_'+predicting+'_'+method+'.csv'
        perf_dict = {'k': [k], 'kmer': [kmer], 'predicting': [predicting], 'method': [method], 'training_time': [training_time], 'max_training_memory': [max_training_memory]}
        df = pd.DataFrame(perf_dict)
        df.to_csv(performance_metrics_results_path, header=True)

        return [training_time, max_training_memory]