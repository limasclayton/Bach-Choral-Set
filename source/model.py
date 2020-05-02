# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ml
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
# paths
dataset_path = "input/bach_choral_set_dataset.csv"

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path, index_col=['event_number'])
X = df.drop('chord_label', axis=1)
y = df.chord_label

# preprocessing
X = X[['bass','meter']]
X = pd.get_dummies(X, drop_first=True)

# feature engeniring

# model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

# Logistic Regression
lr = LogisticRegression(random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
print('-' * 100)
print('LR train score: {:.3f}'.format(lr.score(X_train, y_train)))
print('LR test score: {:.3f}'.format(lr.score(X_test, y_test)))