import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


filepath = os.getcwd() + '/data/'

class results_gen():
    def __init__(self, kmer, k, predicting):
        self.kmer = kmer
        self.k = k
        self.predicting = predicting
        self.X_test = None
        self.y_test = None
        self.model = None

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
        ## ovo and l2
    def modelTraining(self):
        X_train, y_train = self.dataImport()
        kmer = self.kmer
        k = self.k
        predicting = self.predicting
        method='l2'
        
        model = LogisticRegression(max_iter = 1000) # default is l2
        ovo = OneVsOneClassifier(model)
        LRparam_grid = {
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
        ovo_tuned = GridSearchCV(estimator = ovo, param_grid = LRparam_grid, refit=True, cv=10, verbose=1, scoring='balanced_accuracy', n_jobs = -1)
        ovo_tuned.fit(X_train, y_train)
        self.model = ovo_tuned
        return ovo_tuned 

    def modelTesting(self):
        kmer = self.kmer
        k = self.k
        predicting = self.predicting
        method='l2'
        ovo_tuned = self.modelTraining()
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

#case = results_gen('all', '17', 'absence')
#case.modelTraining()