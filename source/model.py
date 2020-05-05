# IMPORTS
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
from xgboost import XGBClassifier

# paths
dataset_path = "input/bach_choral_set_dataset.csv"

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path, index_col=['event_number'])
X = df.drop('chord_label', axis=1)
y = df.chord_label

# PREPROCESSNG
X.drop('choral_ID', axis=1, inplace=True)
X = pd.get_dummies(X, drop_first=True)
X.to_csv('x.csv')

# FEATURE ENGENIRING
# Transform meter in categorical variable
# FEATURE SELECTION

# MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# XGBoost
param_distributions_xgb = {
    'learning_rate' : np.linspace(0, 1, 50),
    'min_split_loss' : np.logspace(1, 3, 10),
    'max_depth' : np.arange(2, 10, 1),
    'min_child_weight' : np.arange(0, 5, 1),
    'subsample' : np.linspace(0.01, 1, 10),
    'colsample_bytree' : np.linspace(0.01, 1, 10),
    'colsample_bylevel' : np.linspace(0.01, 1, 10),
    'colsample_bynode' : np.linspace(0.01, 1, 10)
}

xgb = XGBClassifier()
xgb_CV = RandomizedSearchCV(xgb, param_distributions=param_distributions_xgb, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
xgb_CV.fit(X_train, y_train)
print('-' * 100)
print('XGB train score: {:.3f}'.format(xgb_CV.score(X_train, y_train)))
print('XGB test score: {:.3f}'.format(xgb_CV.score(X_test, y_test)))
print('XGB best params: {0}'.format(xgb_CV.best_params_))


# logistic pipeline
param_grid = [
    {'classifier__C' : np.logspace(-4, 1, 25),
    'classifier__max_iter' : np.arange(300, 400, 25),
    'classifier__multi_class' : ['multinomial']}
    ]

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
    ])

#LeaveOneOut()
lr_cv = RandomizedSearchCV(lr_pipe, param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
lr_cv.fit(X_train, y_train)

print('-' * 100)
print('Pipeline train score: {:.3f}'.format(lr_cv.score(X_train, y_train)))
print('Pipeline test score: {:.3f}'.format(lr_cv.score(X_test, y_test)))
print('Pipeline Best score: {0}'.format(lr_cv.best_score_))
print('Pipeline best params: {0}'.format(lr_cv.best_params_))