from __future__ import division
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

#out_dir = "/mnt/c/Users/xSix.SixAxiS/Documents/Stanford/Spring 2019/CS 221/project/cs221project"
out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"
filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

# split into same train and test sets as others using sklearn package
X_train, X_test, y_train, y_test = train_test_split(admissions, admissions['HOSPITAL_EXPIRE_FLAG'], test_size=0.2, random_state=0)

y_pred = y_test.value_counts().idxmax()
total = len(y_test.index)
correct = y_test.value_counts().max()
y_pred = pd.Series([1] + [y_pred] * len(y_test - 1))
#y_pred = pd.Series([y_pred] * len(y_test - 1))
df_confusion = pd.crosstab(y_test, y_pred) 
df_norm = df_confusion.values / df_confusion.sum(axis=1)[:,None]
ax = sn.heatmap(df_confusion, annot=True, annot_kws={"size": 20}, cmap="YlGnBu")
plt.xlabel('Predicted label', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.title('Confusion Matrix w/o Normalization (baseline)', fontsize=20)
plt.show()
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
score = correct / total
average_precision = metrics.average_precision_score(y_test, y_pred)
auc = metrics.auc(recall, precision)
print("Majority predicts {}, score: {}".format(y_pred[0], score)) 
print("Average PR score: {0:0.2f}".format(average_precision))
print("AUC: {0:0.2f}".format(auc))
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.title('Precision-Recall Curve Baseline', fontsize=20)
#plt.show()
