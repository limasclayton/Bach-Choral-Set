# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ml
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, LeaveOneOut
from sklearn.linear_model import LogisticRegression
# paths
dataset_path = "input/bach_choral_set_dataset.csv"

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path, index_col=['event_number'])
X = df.drop('chord_label', axis=1)
y = df.chord_label

# preprocessing
X.drop('choral_ID', axis=1, inplace=True)
X = pd.get_dummies(X, drop_first=True)

# feature engeniring

# model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratity=y, test_size=0.3, random_state=RANDOM_STATE)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# pipeline
param_grid = [
    {'classifier__C' : np.logspace(-4, 1, 25),
    'classifier__max_iter' : np.arange(300, 400, 25),
    'classifier__multi_class' : ['multinomial']}
    ]

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=RANDOM_STATE))
    ])
#LeaveOneOut()
random = RandomizedSearchCV(pipe, param_grid, cv=5, random_state=RANDOM_STATE)
random.fit(X_train, y_train)

print('-' * 100)
print('Pipeline train score: {:.3f}'.format(random.score(X_train, y_train)))
print('Pipeline test score: {:.3f}'.format(random.score(X_test, y_test)))
print('Pipeline best params: {0}'.format(random.best_params_))