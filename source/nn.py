import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy

# paths
dataset_path = "input/bach_choral_set_dataset.csv"

# variables
RANDOM_STATE = 123
df = pd.read_csv(dataset_path, index_col=['event_number'])

# PREPROCESS

# Dropping choral_ID duo to many ids
df.drop('choral_ID', axis=1, inplace=True)

# Labelling target feature
df.chord_label = LabelEncoder().fit_transform(df.chord_label)

# FEATURE SELECTION
X = df.drop('chord_label', axis=1)
y = df.chord_label

X_dummies = pd.get_dummies(X, columns=['meter'])
X_dummies = pd.get_dummies(X_dummies, drop_first=True)
print(X_dummies.info())

X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=RANDOM_STATE)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(len(np.unique(y)))

# model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(102, activation='softmax'))
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

callbacks = EarlyStopping(monitor='val_loss', patience=25)
history = model.fit(X_train, y_train, epochs=500, validation_split=0.1, callbacks=[callbacks])

# Plotting train history for mse
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plotting train history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plotting train history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluating the model
evaluation = model.evaluate(X_test, y_test)
print('Test loss:', evaluation[0])
print('Test acc:', evaluation[1])