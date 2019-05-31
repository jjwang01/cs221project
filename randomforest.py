import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"

filename = "DIAGNOSES_ICD.csv"
diagnoses_icd = pd.read_csv("{}/{}".format(out_dir, filename))

filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

merged = pd.merge(diagnoses_icd, admissions, on='SUBJECT_ID')
y = merged['HOSPITAL_EXPIRE_FLAG']
merged.drop(columns=['HOSPITAL_EXPIRE_FLAG'])

onehot_encoder = OneHotEncoder(sparse=True)
X = onehot_encoder.fit_transform(merged.astype(str).to_numpy().reshape(-1,len(merged.columns)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#sc = StandardScaler()  
#X_train = sc.fit_transform(X_train)  
#X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=3, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
