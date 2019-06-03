import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import datetime as dt
import sys

# TODO: add gzip code

# flags: 
# ICD9: '-i'
# ETHNICITY: '-e'
# ICD9 and ETHNICITY: '-i -e'
# ICU stay: '-c'
# ED stay: '-d'
# all: '-a'
user_args = sys.argv[1:]
print(user_args)

pd.options.mode.chained_assignment = None
# PREPROCESSING
# put your directory here
out_dir = "/mnt/c/Users/xSix.SixAxiS/Documents/Stanford/Spring 2019/CS 221/project/cs221project"
filename = "DIAGNOSES_ICD.csv"
diagnoses_icd = pd.read_csv("{}/{}".format(out_dir, filename))

filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

# both diagnoses_icd and admissions have SUBJECT_ID as a column
# need to merge these together
result = pd.merge(diagnoses_icd, admissions, on='SUBJECT_ID')
print(result['HOSPITAL_EXPIRE_FLAG'].value_counts())

# use linear regression on numpy arrays to try to predict the expire flag
# one-hot encode ICD9_CODE b/c it's a categorical variable, avoid misrepresenting data
onehot_encoder = preprocessing.OneHotEncoder(sparse=True)
X = [] 
if '-i' in user_args and '-e' in user_args:
    print("ICD_9 and ethnicity")
    X = onehot_encoder.fit_transform(result[['ICD9_CODE', 'ETHNICITY']].astype(str).to_numpy().reshape(-1,2))
    y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
elif '-i' in user_args:
    print("ICD_9")
    X = onehot_encoder.fit_transform(result['ICD9_CODE'].astype(str).to_numpy().reshape(-1,1))
    y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
elif '-e' in user_args:
    print("ethnicity")
    X = onehot_encoder.fit_transform(result['ETHNICITY'].astype(str).to_numpy().reshape(-1,1))
    y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
elif '-d' in user_args:
    print("ED stay")

    # PREPROCESSING
    admissions = admissions[['EDREGTIME', 'EDOUTTIME', 'HOSPITAL_EXPIRE_FLAG']]
    admissions = admissions.dropna()
    # each time is a date and time stamp
    # need to use timedelta in order to get time differences
    X = admissions[['EDREGTIME', 'EDOUTTIME']]
    X['timedelta'] = pd.to_datetime(admissions['EDOUTTIME'], infer_datetime_format=True) - pd.to_datetime(admissions['EDREGTIME'], infer_datetime_format=True)
    X['timedelta'] = X['timedelta'].dt.days
    X = X['timedelta'].to_numpy().reshape(-1,1)
    y = admissions['HOSPITAL_EXPIRE_FLAG'].to_numpy()
elif '-c' in user_args:
    print("ICU stay")

    # each time is a date and time stamp
    # need to use timedelta in order to get time differences
    X = admissions[['ADMITTIME', 'DISCHTIME']]
    X['timedelta'] = pd.to_datetime(admissions['DISCHTIME'], infer_datetime_format=True) - pd.to_datetime(admissions['ADMITTIME'], infer_datetime_format=True)
    X['timedelta'] = X['timedelta'].dt.total_seconds()
    X = X['timedelta'].to_numpy().reshape(-1,1)
    y = admissions['HOSPITAL_EXPIRE_FLAG'].to_numpy()
elif '-a' in user_args:
    print("all")
    y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
    result = result.drop(columns=['HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS'])
    X = onehot_encoder.fit_transform(result.astype(str).to_numpy().reshape(-1,len(result.columns)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MODELING and LEARNING
reg = LogisticRegression(solver='lbfgs',
    max_iter=500).fit(X_train, y_train)

# INFERENCE
y_pred = reg.predict(X_test)

print('Train Score:', reg.score(X_train, y_train))
print('Score:', reg.score(X_test, y_test))

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

# should also use AUROC -- read up on what this means

