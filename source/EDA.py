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
# We can see from info() that there are no missing values on the dataset
print(df.info())
print(df.head())
print(df.tail())
print(df.describe())

# Analize corr between features and target
# Build grafics to analyse higher correlations between features and targets
# How many chorals do we have?
# How many chord_label each choral plays?
# Groupby bass
# Groupby meter
# Correlation between meter and chord_label
# Correlation between bass and chord_label
# How much each pitch contributes to chord_label?
