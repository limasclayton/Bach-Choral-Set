# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ml
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# paths
dataset_path = "input/bach_choral_set_dataset.csv"

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path, index_col=['event_number'])
X = df.drop('chord_label', axis=1)
y = df.chord_label

# PREPROCESSING
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X.meter = X.meter.astype('category')
X.drop('choral_ID', axis=1, inplace=True)
X_dummies = pd.get_dummies(X, drop_first=True)
print(X_dummies.info())
#X.to_csv('x.csv')

# FEATURE SELECTION
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=RANDOM_STATE)

# MODEL

# Support Vector Machine
param_distributions_svm = {
    'C' : np.logspace(-3, 1, 10),
    'gamma' : ['scale', 'auto'],
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'random_state' : [RANDOM_STATE]
}
svm = SVC()
svm_CV = RandomizedSearchCV(svm, param_distributions=param_distributions_svm, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
svm_CV.fit(X_train, y_train)
print('-' * 100)
print('SVM train score: {:.3f}'.format(svm_CV.score(X_train, y_train)))
print('SMV best validation score: {0}'.format(svm_CV.best_score_))
print('SVM best params: {0}'.format(svm_CV.best_params_))
print('SVM test score: {:.3f}'.format(svm_CV.score(X_test, y_test)))

# logistic pipeline
lr_pipe = Pipeline([
    ('classifier', LogisticRegression())
    ])

param_grid = [
    {'classifier__C' : np.logspace(-4, 1, 25),
    'classifier__max_iter' : np.arange(300, 400, 25),
    'classifier__multi_class' : ['multinomial'],
    'classifier__random_state' : [RANDOM_STATE]}
    ]

lr_cv = RandomizedSearchCV(lr_pipe, param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
lr_cv.fit(X_train, y_train)

print('-' * 100)
print('Logistic Regression pipeline train score: {:.3f}'.format(lr_cv.score(X_train, y_train)))
print('Logistic Regression pipeline test score: {:.3f}'.format(lr_cv.score(X_test, y_test)))
print('Logistic Regression pipelinepeline Best score: {0}'.format(lr_cv.best_score_))
print('Logistic Regression pipeline best params: {0}'.format(lr_cv.best_params_))
print('Logistic Regression pipeline coeficients: {0}'.format(lr_cv.best_estimator_.named_steps['classifier'].coef_))

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
print('XGB best validation score: {0}'.format(xgb_CV.best_score_))
print('XGB test score: {:.3f}'.format(xgb_CV.score(X_test, y_test)))
print('XGB best params: {0}'.format(xgb_CV.best_params_))


# Catboost
cat_features = [
    'pitch_1',
    'pitch_2',
    'pitch_3',
    'pitch_4',
    'pitch_5',
    'pitch_6',
    'pitch_7',
    'pitch_8',
    'pitch_9',
    'pitch_10',
    'pitch_11',
    'pitch_12',
    'bass',
    'meter'
]

cat = CatBoostClassifier(cat_features=cat_features, random_seed=RANDOM_STATE)
cat.fit(X_train_cat, y_train_cat)
print('CatBoost train score: {:.3f}'.format(cat.score(X_train_cat, y_train_cat)))
print('CatBoost test score: {:.3f}'.format(cat.score(X_test_cat, y_test_cat)))
