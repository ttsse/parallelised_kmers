import pandas as pd
import os

filepath = os.getcwd() + '/data/'

def results_gen(kmer, k, predicting):

    x_train_path = filepath+'10ribo_X_train_readfile_by_'+kmer+'_'+predicting+'_df_k'+k+'.csv'
    x_test_path = filepath+'10ribo_X_test_readfile_by_'+kmer+'_'+predicting+'_df_k'+k+'.csv'
    y_train_path = filepath+'10ribo_y_train_readfile_by_all_kmer_trueribotype_df_k'+k+'.csv'
    y_test_path = filepath+'10ribo_y_test_readfile_by_all_kmer_trueribotype_df_k'+k+'.csv'

    try:
        X_train = pd.read_csv(x_train_path, index_col=0)
        X_test= pd.read_csv(x_test_path, index_col=0)
        y_train = pd.read_csv(y_train_path, index_col=0)
        y_test = pd.read_csv(y_test_path, index_col=0)

    except FileNotFoundError as error:
        raise FileNotFoundError(error)


