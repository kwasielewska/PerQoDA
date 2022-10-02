#!/usr/bin/env python3
import warnings
import weles as ws
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from multiprocessing import Pool, Process, Manager, current_process
import dataframe_image as dfi
import yaml
from yaml.loader import SafeLoader
from pandas import read_csv
from progress.bar import Bar
from progressbar import progressbar
from progress.spinner import MoonSpinner
import os


class PerQoDA:

    def __init__(self):
        self.raw_dataset = None
        self.datasets = None
        self.clfs = None
        self.clfs_ver1 = []
        self.clfs_ver2 = {}
        self.ev = None
        self.X1 = None
        self.y1 = None
        self.dataset_params = {}
        self.a = None
        self.eval_metrics = {}
        self.metrics = {}
        self.perc = None
        self.perm = None
        self.corr = None
        self.nperm = None
        self.output = None

    # Parse and load the configuration file
    # TODO define additional values for static parameters - p-value treshold
    def loadConfig(self):
        config = None
        try:
            with open('config.txt') as f:
                config = yaml.load(f, Loader=SafeLoader)
        except Exception as err:
            print("Error: Unable to read the configuration file. Please check formating or file access.")
            print("Full Error Message",err)

        self.filename = config["dataset"] 
        self.label = config["dataset_label"]
        self.clfs = config["classifiers"]
        self.eval_metrics = config["metrics"]
        self.verbose = int(config["verbose_level"])
        self.nperm = config["permutations"]
        self.perc = config["percentages"]
        self.delimiter = config["delimiter"]
        self.cores = config["cores"]
        self.output = config["output"]
        
        # Disable debug messages for lower verbose levels (1,2)
        if self.verbose <= 1:
            np.seterr(all="ignore")
            warnings.filterwarnings("ignore")
            warnings.simplefilter('ignore', np.RankWarning)
    def checkOutput(self):
        try:
            isdir = os.path.isdir(self.output)
            if isdir == False:
                if self.verbose >= 1:
                    print("Unbable to find output directory. Creating a new one.")
                os.mkdir(self.output)
            else:
                if self.verbose >= 1:
                    print("Output directory already exists. ")
        except Exception as err:
            print("Error: Unable to create or access output directory in path",self.ouput)
            print("Full Error Message",err)

    # Load dataset
    def loadDataset(self):
        try:
            self.raw_dataset = pd.read_csv(filepath_or_buffer=self.filename, sep=self.delimiter)
        except Exception as err:
            print("Error: Unable to read input dataset. Please check formating or file access.")
            print("Full Error Message",err)

        try:
            self.y1 = self.raw_dataset[self.label]
            self.X1 = self.raw_dataset.drop(columns=[self.label])
            self.X1 = MinMaxScaler().fit_transform(self.X1)
        except ValueError as err:
            print("Error: convert string value to number.")
            print("Full Error Message:",err)

        self.datasets = {
            "all": (self.X1, self.y1)
        }

        # Calculate dataset statistics
        total_negative_samples = (self.y1 == 0).sum()
        total_positive_samples = (self.y1 == 1).sum()
        ratio = total_negative_samples / total_positive_samples
        self.dataset_params["ratio"] = ratio
        N = self.y1.shape[0]
        prevalence = total_positive_samples / N

        # Print dataset statistics if needed
        if self.verbose >= 1:
            print('Number of obs:',N)
            print('Number of positives:',total_positive_samples)
            print('Number of negatives:',total_negative_samples)
            print('Prevalence:',prevalence)
            print("Class Ratio:",ratio)

    # Initialize listed pool of classifiers
    # TODO handle unknown or duplicit models defined in configuration files
    def runClassifiers(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.neural_network import MLPClassifier

        # Pool of available models for selected pool of classifiers
        clfs_pool = {
                "KNN": KNeighborsClassifier(),
                "DT": DecisionTreeClassifier(),
                "RF": RandomForestClassifier(),
                "MLP": MLPClassifier(hidden_layer_sizes=(80,100), activation='relu', batch_size=20, max_iter=200, verbose=0),
                "AB": AdaBoostClassifier(),
                "XGB": XGBClassifier(use_label_encoder=False, scale_pos_weight=self.dataset_params["ratio"])
        }
        
        for classifier in self.clfs:
            self.clfs_ver1.append({classifier: clfs_pool[classifier]})
            self.clfs_ver2[classifier] = clfs_pool[classifier]


    # Initialize metrics and set true values
    def runMetrics(self):
        from sklearn.metrics import precision_score, f1_score, balanced_accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score, accuracy_score, fbeta_score, recall_score
        from imblearn.metrics import sensitivity_score, specificity_score

        self.ev = ws.evaluation.Evaluator(datasets=self.datasets, protocol2=(False, 2, None)).process(clfs=self.clfs_ver2, verbose=0)

        def true_positive_rate(y_true, y_pred):
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            return tp / (tp + fn)

        def false_positive_rate(y_true, y_pred):
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            return fp / (fp + tn)

        def precision_from_tpr_fpr(y_true, y_pred):
            #positive class prevalence
            self.y1 = self.datasets["all"][1]
            count1 = (self.y1 == 1).sum()
            N = self.y1.shape[0]
            prevalence = count1 / N

            if ((true_positive_rate(y_true, y_pred) == 0) & (false_positive_rate(y_true, y_pred) == 0)):
                print("0/0 case")
                return 0
            else:
                return (prevalence * true_positive_rate(y_true, y_pred)) / (prevalence * true_positive_rate(y_true, y_pred) + ((1 - prevalence) * false_positive_rate(y_true, y_pred)))

        def F2_score(y_true, y_pred):
            return fbeta_score(y_true, y_pred, beta=2)

        metrics_pool = {
            "precision": precision_score,
            "recall": recall_score,
            "sensitivity/recall": sensitivity_score,
            "F1": f1_score,
            "BAc": balanced_accuracy_score,
            "AP": average_precision_score,
            "specificity": specificity_score,
            "MCC": matthews_corrcoef,
            "ROC": roc_auc_score,
            "Acc": accuracy_score,
            "FPR": false_positive_rate,
            "TPR": true_positive_rate,
            "PTF": precision_from_tpr_fpr,
            "F2": F2_score
        }

        for metric in self.eval_metrics:
            self.metrics[metric] = metrics_pool[metric]

        scores = self.ev.score(metrics=self.metrics)

    # Run permutation tests to evalute the quality
    def permutation(self):
        self.a=np.shape(self.ev.scores.mean(axis=2)[:, :, 0]) # true result
        self.perm = np.zeros((self.nperm,len(self.perc),self.a[1]))
        self.corr = np.zeros((self.nperm,len(self.perc)))

        # Main testing loop
        # TODO improve paralel processing - each clasifier will be evaluted completely separately
        with Bar('Evaluating Dataset Quality...',max=self.nperm*len(self.perc)) as bar:
            for i in range(self.nperm):
                for j in range(len(self.perc)):
                    if self.verbose >= 1:
                        print("iteration",i,j)
                    t=0
                    while True:
                        ind1=np.where(self.y1 == 0)
                        ind2=np.where(self.y1 == 1)
                    
                        nperc1 = round(self.perc[j]*len(ind1[0])/100)
                        nperc2 = round(self.perc[j]*len(ind2[0])/100)
                    
                        indP = np.random.permutation(np.concatenate((ind1[0][:nperc1], ind2[0][:nperc2])))
                        ind = np.sort(indP);

                        y1P = np.copy(self.y1);

                        y1P[ind] = self.y1[indP];
                        
                        comparison = self.y1 == y1P
                        
                        if not comparison.all() or t > 3:
                            if self.verbose >= 1:
                                print("Too many permutations with the same result. Skipping this iteration...")
                                print("Note: This usually hapends for small or suspicious datasets.")
                            break
                        t += 1

                    self.datasetsP = {
                    "all": (self.X1, y1P)
                    }

                    ## Non-paralel version of classifier evaluation
                    #evP = ws.evaluation.Evaluator(datasets=self.datasetsP,protocol2=(True, 2, None)).process(clfs=self.clfs_ver2, verbose=0)
                    #scores = evP.score(metrics=self.metrics)
                    #self.perm[i,j,:] = evP.scores.mean(axis=2)[:, :, 0]
                    #kk = np.corrcoef(y1P,self.y1)
                    #self.corr[i,j] = kk[0,1]
                    
                    ## Paralel version of classifier evaluation
                    with Pool(self.cores) as p:
                        eval_scores = p.map(self.evaluate, self.clfs_ver1)
                    tmp_scores = np.zeros((1, len(self.clfs_ver1)))
                    for idx, item in enumerate(eval_scores):
                        tmp_scores[0][idx] = item[0]
                    self.perm[i,j,:] = tmp_scores 
                    kk = np.corrcoef(y1P,self.y1)
                    self.corr[i,j] = kk[0,1]

                    bar.next()
                    
    # Helper function for parallel processing
    def evaluate(self,classifier):
        evP2 = ws.evaluation.Evaluator(datasets=self.datasetsP,protocol2=(True, 2, None)).process(clfs=classifier, verbose=0)
        scores = evP2.score(metrics=self.metrics)
        return evP2.scores.mean(axis=2)[:, :, 0][0]

    # Generate output report files for dataset quality
    def printResults(self):
        import matplotlib.cm as cm
        classifiers = ()
        for i in self.clfs_ver2:
            classifiers = classifiers + (i,)
        pvalues = np.zeros((self.a[1],len(self.perc)))

        colors = cm.rainbow(np.linspace(0, 1, self.a[1]))
        # Plot true values as diamonds
        for i, c in zip(range(self.a[1]),colors):
            plt.scatter(1.1+i*0.01, self.ev.scores.mean(axis=2)[:, i, 0], s=100, color=c, marker='d')

        plt.legend(classifiers, prop={'size': 8})
 
        # Plot lines for true values
        for i, c in zip(range(self.a[1]),colors):
            plt.plot([0, 1.1+i*0.01], [self.ev.scores.mean(axis=2)[:, i, 0], self.ev.scores.mean(axis=2)[:, i, 0]], c=c, linestyle='dashed', alpha=0.5)

        # Plot permutations
        colors = cm.rainbow(np.linspace(0, 1, self.a[1]))
        for j in range(len(self.perc)):
            for i, c in zip(range(self.a[1]),colors):
                ind = np.where(self.perm[:,j,i] < self.ev.scores.mean(axis=2)[:, i, 0])
                plt.scatter((self.corr[ind,j]), self.perm[ind,j,i], color="none", edgecolor=c, alpha=0.3)
                
        for j in range(len(self.perc)):
            for i, c in zip(range(self.a[1]),colors):
                ind = np.where(self.perm[:,j,i] >= self.ev.scores.mean(axis=2)[:, i, 0])
                plt.scatter((self.corr[ind,j]), self.perm[ind,j,i], color=c, edgecolor="black", alpha=1)
                pvalues[i,j] = ((len(ind[0])+1)*1.0)/(self.nperm+1);

        plt.ylabel('Performance (Recall)', size=12)
        plt.xlabel('Permutation Correlation', size=12)       

        plt.plot([0, 1.1], [self.perm.min(), self.perm.min()], color='red', linestyle='dashed', alpha=0.5)

        plt.axis([-0.05, 1.2, 0, 1.1])

        # Save permutation chart
        plt.savefig(self.output+"/permutation-chart.png")

        pv = pd.DataFrame(data=pvalues, index=list(classifiers), columns=self.perc)

        def significant(v):
            return "font-weight: bold; color: red" if v > 0.01 else None

        # Save p-value table
        pv.style.applymap(significant)
        dfi.export(pv, self.output+"/pvalues.png")

        # Get slope chart
        names = classifiers
        cor = []
        per = []
        slopes = []

        for i, c in zip(range(self.a[1]),colors):
            for j in range(len(self.perc)):
                plt.scatter(np.mean(self.corr[:,j]), np.mean(self.perm[:,j,i]), color=c, alpha=1)
            
            cor = np.mean(self.corr[:,:], axis=0)
            per = np.mean(self.perm[:,:,i], axis=0) 

            
            slope, intercept = np.polyfit(cor, per, 1)
            plt.plot(cor, slope*cor + intercept, color=c, linewidth=0.8)
            
            if self.verbose >= 1:
                print(names[i], '=', slope)
            slopes = np.append(slopes, slope)

        plt.legend(names, prop={'size': 8})
        plt.savefig(self.output+"/slope-chart.png")
        maxind = np.argmax(abs(slopes))
        if self.verbose >= 1:
            print('Slope:', np.max(abs(slopes)), '-', names[maxind])

    # Wrapper function to load configuration file and dataset
    def load(self):
        with MoonSpinner('Initializing Configuration...') as bar:
            bar.next()
            self.loadConfig()
            bar.next()
            self.loadDataset()
            bar.next()
    
    # Wrapper function to initialize classifiers and metrics
    def prepareRun(self):
        with MoonSpinner('Initializing Classifiers and Merics...') as bar:
            bar.next()
            self.runClassifiers()
            bar.next()
            self.runMetrics()
            bar.next()

    # Wrapper fuction for permutation testing and generating report
    def run(self):
        self.permutation()
        self.printResults()

# Main
if __name__ == "__main__":
    qod = PerQoDA()
    qod.load()
    qod.prepareRun()
    qod.run()
    print("Evaluation Completed! You find your results in "+qod.output+" directory.")

 