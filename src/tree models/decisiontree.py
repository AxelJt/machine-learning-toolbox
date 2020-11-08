from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
from plotly import tools
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import plot_tree, export_text

import numpy as np
import pandas as pd


class SuperDTC(DecisionTreeClassifier):

    def __init__(self, X_train, X_test, y_train, y_test, params):
        
        super().__init__(**params)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        super().fit(self.X_train, self.y_train)

    def predict(self, X=None):
        if X is not None:
            return super().predict(X)
        else:
            return super().predict(self.X_test)

    def predict_proba(self, X=None):
        if X is not None:
            return super().predict_proba(X)[:,1]
        else:
            return super().predict_proba(X_test)[:,1]

    def get_confusion_matrix(self):
        
        tn, fp, fn, tp = confusion_matrix(
            self.y_test, 
            self.predict()
        ).ravel()

        return {
            'true_neg':tn,
            'false_pos':fp,
            'false_neg':fn,
            'true_pos':tp
        }

    def get_sensitivity(self):
        res = self.get_confusion_matrix()
        return 100.*res['true_pos'] / (res['true_pos']+res['false_neg'])
    
    def get_specificity(self):
        res = self.get_confusion_matrix()
        return 100.*res['true_neg'] / (res['true_neg']+res['false_pos'])

    def get_precision(self):
        res = self.get_confusion_matrix()
        return 100.*res['true_pos'] / (res['true_pos']+res['false_pos'])
    
    def get_f1(self):
        pre, rec = self.get_precision(), self.get_sensitivity()
        return 2*(pre*rec)/(pre+rec)

    def get_tree(self, features, class_names=None):

        plt.figure(figsize=(60,60))
        plot_tree(
            self,
            feature_names = features,
            filled=True, 
            proportion = True,
            class_names=class_names
        )
        plt.show()

    def get_roc(self):

        dt_fpr, dt_tpr, dt_thresholds = roc_curve(
            self.y_test, 
            self.predict_proba(self.X_test)
        )

        random_curve = Scatter(
            x = [0,1], 
            y = [0,1], 
            mode = 'lines', 
            line=dict(dash = 'dot'), 
            name = 'Random classifier', 
            marker = dict(color = '#db2c24')
        )

        dt_roc_curve = Scatter(
            x = dt_fpr, 
            y = dt_tpr, 
            name = 'Decision Tree - AUC : {}'.format(round(auc(dt_fpr, dt_tpr),2)), 
            mode = 'lines', 
            marker = dict(color = '#cc9aa3'), 
            text=["Seuil : {}".format(round(x,2)) for x in dt_thresholds]
        )

        layout=Layout(
            title='ROC curve',
            xaxis=dict(title='False Positive Rate (FP/N)'),
            yaxis=dict(title='True Positive Rate (TP/P)')
        )

        iplot(Figure(data = [dt_roc_curve, random_curve], layout=layout))
