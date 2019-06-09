from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

out_dir = "/mnt/c/Users/xSix.SixAxiS/Documents/Stanford/Spring 2019/CS 221/project/cs221project"
filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

# split into same train and test sets as others using sklearn package
X_train, X_test, y_train, y_test = train_test_split(admissions, admissions['HOSPITAL_EXPIRE_FLAG'], test_size=0.2, random_state=0)

y_pred = y_test.value_counts().idxmax()
total = len(y_test.index)
correct = y_test.value_counts().max()

score = correct / total
print("Majority predicts {}, score: {}".format(y_pred, score)) 

