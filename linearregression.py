import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# TODO: add gzip code

# PREPROCESSING
out_dir = "/home/justin_jiayi_wang/cs221project"
filename = "DIAGNOSES_ICD.csv"
diagnoses_icd = pd.read_csv("{}/{}".format(out_dir, filename))

filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

# both diagnoses_icd and admissions have SUBJECT_ID as a column
# need to merge these together
result = pd.merge(diagnoses_icd, admissions[['SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG']], on='SUBJECT_ID')

# use linear regression on numpy arrays to try to predict the expire flag
# one-hot encode ICD9_CODE b/c it's a categorical variable, avoid misrepresenting data
label_encoder = preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(result['ICD9_CODE'].astype(str).to_numpy())
onehot_encoder = preprocessing.OneHotEncoder(sparse=True)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(onehot_encoded, y, test_size=0.2, random_state=0)

print(X_test.shape, y_test.shape)
print(X_test[:,0].flatten().shape, y_test.shape)

# MODELING and LEARNING
reg = LinearRegression().fit(X_train, y_train)

# INFERENCE
y_pred = reg.predict(X_test)

plt.scatter(X_test[:,0].flatten(), y_test,  color='gray')
plt.plot(X_test[:,0].flatten(), y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
