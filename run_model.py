import subprocess 
from multiprocessing.dummy import Pool


#kmer_list = ['all', 'kmer']
#k_list = ['17', '21', '31']
#predicting_list = ['absence', 'sumcount', 'meancount', 'mediancount']

kmer_list = ['all']
k_list = ['17']
predicting_list = ['absence']

command_list = [f"import os; os.chdir('/Users/vaish555/Documents/Uppsala Universitet/(1) courses/(12) TTSSE/ttsse/'); from data_modelling import results_gen; results_gen('{km}','{k}','{p}').modelTraining()" for km in kmer_list for k in k_list for p in predicting_list]

def run_model(x):
    subprocess.call(["python3", "-c", x])

with Pool(len(command_list)) as p:
    p.map(run_model, command_list)