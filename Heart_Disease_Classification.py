import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

for column in df.columns:
    print('=================================')
    print(f"{column} : {df[column].unique()}")

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_score(clf, X_train, y_train, X_test, y_test, train):
    if train == True:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print('Train Result:\n====================================')
        print(f'Accuracy Score: {accuracy_score(y_train, pred) * 100 :.2f} %')
        print('---------------------------------------------------')
        print(f'Classification Report:\n{clf_report}')
        print('---------------------------------------------------')
        print(f'Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n ')

    elif train == False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print('Test Result:\n====================================')
        print(f'Accuracy Score: {accuracy_score(y_test, pred) * 100 :.2f} %')
        print('---------------------------------------------------')
        print(f'Classification Report:\n{clf_report}')
        print('---------------------------------------------------')
        print(f'Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n ')


print('Logistic Regression')
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

print_score(lr, X_train, y_train, X_test, y_test, train=True)
print_score(lr, X_train, y_train, X_test, y_test, train=False)

import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(lr, file)
