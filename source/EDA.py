import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# caminhos
dataset_path = "input/bach_choral_set_dataset.csv"

# lendo
df = pd.read_csv(dataset_path, index_col=['event_number'])

# initial EDA
# info(), head(), tail() e describe() from data
# We can see from info() that there are no missing values on the dataset
print(df.info())
print(df.head())
print(df.tail())
print(df.describe())

# Analize corr between features and target
# meter is the only numeric feature on the dataset, so let's see the other comparations
print(df.corr())

# How many chord_label do we have?
print('\nChord Labels')
print(len(df.chord_label.unique()))
print(df.chord_label.value_counts())

# How many chorals do we have?
print('\nHow many chorals do we have?')
print(len(df.choral_ID.unique()))
print(df.choral_ID.value_counts())

# How many chord_label each choral plays?
print('\nHow many chord_label each choral plays?')
print(df.groupby(['choral_ID','chord_label'])['chord_label'].count())

# Groupby bass
print('\nBass')
print(df.bass.value_counts())

# Groupby meter
print('\nMeter')
print(df.meter.value_counts())

# Correlation between meter and chord_label 
print('\nCorrelation between meter and chord_label')
print(df.groupby(['choral_ID','chord_label'])['chord_label'].count())
df.groupby(['meter','chord_label'])['chord_label'].count().to_csv('meter_chord_label.csv')

# Correlation between bass and chord_label appers to be large
print('\nCorrelation between bass and chord_label')
print(df.groupby(['bass','chord_label'])['chord_label'].count())
df.groupby(['bass','chord_label'])['chord_label'].count().to_csv('bass_chord_label.csv')

# How much each pitch contributes to chord_label?
