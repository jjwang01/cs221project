import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# TODO: add gzip code

# PREPROCESSING
out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"
filename = "DIAGNOSES_ICD.csv"
diagnoses_icd = pd.read_csv("{}/{}".format(out_dir, filename))

filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

# both diagnoses_icd and admissions have SUBJECT_ID as a column
# need to merge these together
result = pd.merge(diagnoses_icd, admissions, on='SUBJECT_ID')
output = result['HOSPITAL_EXPIRE_FLAG']
result.drop(columns=['HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS', 'DEATHTIME', 'DISCHARGE_LOCATION'])
# use linear regression on numpy arrays to try to predict the expire flag
# one-hot encode ICD9_CODE b/c it's a categorical variable, avoid misrepresenting data
onehot_encoder = preprocessing.OneHotEncoder(sparse=True)
onehot_encoded = onehot_encoder.fit_transform(result.astype(str).to_numpy().reshape(-1,len(result.columns)))
y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(onehot_encoded, y, test_size=0.2, random_state=0)

# MODELING and LEARNING
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)

# INFERENCE
y_pred = clf.predict(X_test)

y_train_pred = clf.predict(X_train)
df_confusion = pd.crosstab(y_test, y_pred)
df_norm = df_confusion.values / df_confusion.sum(axis=1)[:,None]

ax = sn.heatmap(df_norm, annot=True, annot_kws={"size": 20}, cmap = "YlGnBu")

plt.xlabel('Predicted label', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.title('Confusion Matrix, w/Normalization (NN)', fontsize=20)
plt.show()
#have to run pca in order for plot to show?
#plt.scatter(X_test[:,0], y_test,  color='gray')
#plt.plot(X_test[:,0], y_pred, color='red', linewidth=2)
#plt.show()

precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
auc = metrics.auc(recall, precision)

print('Score:', clf.score(X_test, y_test))
print('AUC: {0:0.2f}'.format(auc))

plt.plot([0,1],[0.5,0.5],linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve NN')
plt.show()
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
