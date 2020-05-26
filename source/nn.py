import pandas as pd
import numpy as np

# sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# paths
dataset_path = "input/bach_choral_set_dataset.csv"

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path, index_col=['event_number'])
X = df.drop('chord_label', axis=1)
y = df.chord_label

X.meter = StandardScaler().fit_transform(X.meter.values.reshape(-1, 1))
X_dummies = pd.get_dummies(X, drop_first=True)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# FEATURE SELECTION
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=RANDOM_STATE)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(len(np.unique(y)))

# model

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(87,)))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(102, activation='softmax'))
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=20)

model.evaluate(X_test, y_test)