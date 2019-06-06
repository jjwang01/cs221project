import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# PREPROCESSING
out_dir = "/Users/justinwang/Desktop/CS 221/cs221project/allennlp"
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

# add the outputs to the end of each value
processed = processed.add(' ').add(y.astype(str))

# split into train, val, and test sets
train, test = train_test_split(processed, test_size=0.2, random_state=1)
train, val = train_test_split(train, test_size=0.2, random_state=1)

train.to_csv('{}/train.txt'.format(out_dir), header=False, index=False)
val.to_csv('{}/val.txt'.format(out_dir), header=False, index=False)
test.to_csv('{}/test.txt'.format(out_dir), header=False, index=False)