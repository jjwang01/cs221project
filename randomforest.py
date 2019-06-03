import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

out_dir = "/mnt/c/Users/xSix.SixAxiS/Documents/Stanford/Spring 2019/CS 221/project/cs221project"

filename = "DIAGNOSES_ICD.csv"
diagnoses_icd = pd.read_csv("{}/{}".format(out_dir, filename))

filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

merged = pd.merge(diagnoses_icd, admissions, on='SUBJECT_ID')
y = merged['HOSPITAL_EXPIRE_FLAG']
merged.drop(columns=['HOSPITAL_EXPIRE_FLAG'])


onehot_encoder = OneHotEncoder(sparse=True)
X = onehot_encoder.fit_transform(merged['ICD9_CODE'].astype(str).to_numpy().reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#sc = StandardScaler()  
#X_train = sc.fit_transform(X_train)  
#X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=3, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 

print(metrics.confusion_matrix(y_test,y_pred))  
print(metrics.classification_report(y_test,y_pred))  
print(metrics.accuracy_score(y_test, y_pred))

# need to use precision and recall instead to see false-positive rates
average_precision = metrics.average_precision_score(y_test, y_pred)

precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
auc = metrics.auc(recall, precision)


print('Average precision-recall score: {0:0.2f}'.format(average_precision))
print('F1 score: {0:0.2f}'.format(f1))
print('AUC: {0:0.2f}'.format(auc))
plt.plot([0,1], [0.5,0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.show()