import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

#comment this out @justin/gaurab
#out_dir = "/home/emily2h/Documents/cs221project"
out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"

filename = "DIAGNOSES_ICD.csv"
diagnoses_icd = pd.read_csv("{}/{}".format(out_dir, filename))

filename = "ADMISSIONS.csv"
admissions = pd.read_csv("{}/{}".format(out_dir, filename))

merged = pd.merge(diagnoses_icd, admissions, on='SUBJECT_ID')
y = merged['HOSPITAL_EXPIRE_FLAG']
merged.drop(columns=['HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS', 'DEATHTIME', 'DISCHARGE_LOCATION'])


onehot_encoder = OneHotEncoder(sparse=True)
X = onehot_encoder.fit_transform(merged['ICD9_CODE'].astype(str).to_numpy().reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#sc = StandardScaler()  
#X_train = sc.fit_transform(X_train)  
#X_test = sc.transform(X_test)

rf = RandomForestClassifier(n_estimators=125, random_state=0, n_jobs=-1)  
rf.fit(X_train, y_train)  
y_pred = rf.predict(X_test) 
print(y_pred[0:5])

df_confusion = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames=['Predicted'])
df_norm = df_confusion.values / df_confusion.sum(axis=1)[:,None]

#ax = sn.heatmap(df_confusion, annot=True, cmap="YlGnBu", annot_kws={"size": 20})
ax = sn.heatmap(df_norm, annot=True, annot_kws={"size": 20}, cmap="YlGnBu")
plt.xlabel('Predicted label', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.title('Confusion Matrix, w/Normalization', fontsize=20)
#plt.title("Confusion Matrix, w/o Normalization", fontsize=20)
plt.show()

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
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve of Random Forests')
plt.show()
