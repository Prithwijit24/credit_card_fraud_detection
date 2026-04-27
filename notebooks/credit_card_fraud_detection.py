# %matplotlib qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', '{:,.2f}'.format)



##data loading
train_data = pd.read_csv('/home/prithwijit/programming/python/fraud_detection/data/fraudTrain.csv')
test_data = pd.read_csv('/home/prithwijit/programming/python/fraud_detection/data/fraudTest.csv')

## data cleaning
train_data.isna().sum()
test_data.isna().sum()
## >>> There are no missing values in both train and test data.

## looking into the data (on different columns)
train_data[['amt']].describe().T
train_data.groupby('is_fraud')[['amt']].describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])

plt.plot(train_data['trans_date_trans_time'], train_data['amt'])
plt.show()
