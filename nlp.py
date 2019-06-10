import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


# PREPROCESSING
#out_dir = "/mnt/c/Users/xSix.SixAxiS/Documents/Stanford/Spring 2019/CS 221/project/cs221project"
out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"
filename = "ADMISSIONS.csv"

# load the dataset
df = pd.read_csv("{}/{}".format(out_dir, filename))
df = df[['DIAGNOSIS', 'HOSPITAL_EXPIRE_FLAG']]
df = df.dropna()

# check class distribution
# print(df['HOSPITAL_EXPIRE_FLAG'].value_counts())
y = df['HOSPITAL_EXPIRE_FLAG']

diagnoses = df['DIAGNOSIS']
# print(diagnoses[:10])

# Remove punctuation
processed = diagnoses.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

# change all to lower case
processed = processed.str.lower()

# remove stop words
stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

# create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

# print the total number of words and the 15 most common words
# print('Number of words: {}'.format(len(all_words)))
# print('Most common words: {}'.format(all_words.most_common(15)))

# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]

# find these 1500 most common words
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Now lets do it for all the messages
messages = list(zip(processed, y))

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# call find_features function for each diagnosis
featuresets = [(find_features(text), label) for (text, label) in messages]

# LEARNING
training, testing = train_test_split(featuresets, test_size=0.2, random_state=seed)

"""
model = SklearnClassifier(SVC(kernel='linear'))

model.train(training)

# INFERENCE
accuracy = nltk.classify.accuracy(model, testing)
"""
######################################################################################

# ONLY RUN ON GOOGLE CLOUD GPU -- TAKES A LONG TIME
# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(solver='lbfgs', max_iter = 500),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))

"""
for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)
    print("{} Accuracy: {}".format(name, accuracy))

    test_true, test_pred = [], []

    for i, (features, label) in enumerate(testing):
        test_true.append(label)
        observed = nltk_model.classify(features)
        test_pred.append(observed)

    # need to use precision and recall instead to see false-positive rates
    average_precision = metrics.average_precision_score(test_true, test_pred)

    precision, recall, thresholds = metrics.precision_recall_curve(test_true, test_pred)
    f1 = metrics.f1_score(test_true, test_pred)
    auc = metrics.auc(recall, precision)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('F1 score: {0:0.2f}'.format(f1))
    print('AUC: {0:0.2f}'.format(auc))
    plt.plot([0,1], [0.5,0.5], linestyle='--')
    plt.plot(recall, precision, marker='.')
    plt.show()
"""
# Ensemble methods = Voting classifier
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble, testing)
print("Voting Classifier: Accuracy: {}".format(accuracy))

# make class label prediction for testing set
txt_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)

# print a confusion matrix and a classification report
print(metrics.classification_report(labels, prediction))

print(pd.DataFrame(
    metrics.confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['not mortality', 'mortality']],
    columns = [['predicted', 'predicted'], ['not mortality', 'mortality']]))
df_confusion = pd.crosstab(labels, prediction) 
df_norm = df_confusion.values / df_confusion.sum(axis=1)[:,None]

ax = sn.heatmap(df_norm, annot=True, annot_kws={"size": 20}, cmap="YlGnBu")
plt.xlabel('Predicted label', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.title('Confusion Matrix, w/Normalization', fontsize=20)
plt.show()


test_true, test_pred = [], []

for i, (features, label) in enumerate(testing):
    test_true.append(label)
    observed = nltk_ensemble.classify(features)
    test_pred.append(observed)

# need to use precision and recall instead to see false-positive rates
average_precision = metrics.average_precision_score(test_true, test_pred)

precision, recall, thresholds = metrics.precision_recall_curve(test_true, test_pred)
f1 = metrics.f1_score(test_true, test_pred)
auc = metrics.auc(recall, precision)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
print('F1 score: {0:0.2f}'.format(f1))
print('AUC: {0:0.2f}'.format(auc))
plt.plot([0,1], [0.5,0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve NLP")
#plt.show()
