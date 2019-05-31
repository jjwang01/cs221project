import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import datetime as dt

# TODO: add gzip code
pd.options.mode.chained_assignment = None

# PREPROCESSING
out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"
filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

# each time is a date and time stamp
# need to use timedelta in order to get time differences
X = admissions[['ADMITTIME', 'DISCHTIME']]
X['timedelta'] = pd.to_datetime(admissions['DISCHTIME'], infer_datetime_format=True) - pd.to_datetime(admissions['ADMITTIME'], infer_datetime_format=True)
X['timedelta'] = X['timedelta'].dt.total_seconds()
X = X['timedelta'].to_numpy().reshape(-1,1)

y = admissions['HOSPITAL_EXPIRE_FLAG'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MODELING and LEARNING
reg = LogisticRegression().fit(X_train, y_train)

# INFERENCE
y_pred = reg.predict(X_test)

#have to run pca in order for plot to show?
#plt.scatter(X_test[:,0], y_test,  color='gray')
#plt.plot(X_test[:,0], y_pred, color='red', linewidth=2)
#plt.show()

print('Score:', reg.score(X_test, y_test))
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
