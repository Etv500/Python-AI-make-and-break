import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

################RATINGS IMPORT#####################
url = (r".csv")
names = names = ['ID', 'Ratings']
df1 = pd.read_csv(url, names=names, header=None, encoding='latin-1')
df1=df1.groupby(['ID']).mean()
df1=df1.reset_index()

################ACTORS/DIRECTORS IMPORT#####################
urll = (r".csv")
names = names = ['ID', 'Actor1','Actor2', 'Actor3', 'Director1']
df2 = pd.read_csv(urll, names = names, header=None, encoding='latin-1')

df = pd.merge(df1, df2, on='ID')
df['Ratings'] = df['Ratings'].astype(int)


################RANDOM FOREST#####################

### One Hot Coding of Actor/director Names and summing up duplicate names
X1 = pd.get_dummies(df[['Actor1', 'Actor2', 'Actor3']], prefix='', prefix_sep='')
X1 = X1.groupby(lambda x:x, axis=1).sum()

X2 = pd.get_dummies(df['Director1'], prefix='', prefix_sep='')
X2 = X2.groupby(lambda x:x, axis=1).sum()

y = df['Ratings']

#Get Train and Test datasets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42)

## Train the models
rfc1 = RandomForestClassifier(random_state=42, n_estimators=1000, verbose=10)
rfc1.fit(X1_train, y1_train)

rfc2 = RandomForestClassifier(random_state=42, n_estimators=1000, verbose=10)
rfc2.fit(X2_train, y2_train)

## Predictions
pred_actors = rfc1.predict(X1_test)
pred_dirs = rfc2.predict(X2_test)


print(classification_report(y1_test, pred_actors))
print(classification_report(y2_test, pred_dirs))
