# Machine-Learning-Chicago-Housing-Market
A machine learning exploration into Chicago, IL house listings, extracted on 05/17/2022. 

Dataset: Cleaned housing listings in Chicago, IL on 05/17/2022 (just a portion, not the full listing data)

A Linear Regression model was built to predict house listing price based on housing features such as zip, number of bedrooms, number of bathrooms, and sq ft.

# Observations:
- Housing list prices in Chicago are not predictable using a linear regression model with the inputs.
- However, the number of bathrooms were slightly predicatable through a linear regression model. Using data that had mising sq ft filled with average sq ft numbers (based on matching number of bathrooms)
the linear regression had an RMSE of 1.33, Relative RMSE of 41.79%, R2 of .455, and an R score of .674 with a P Val of 4.07e^-25. These results show that the error of prediction is 1.33 bedrooms, which is unacceptable for a model that aims to predict the number of bedrooms in a house. However, there appears to be a significant positive relationship between the test data and the model's predictions.
- Looking into the model that drops all null values, instead of filling the null sq ft values, the RMSE actually drops but still remains too high to be accepted (.998), R RMSE of 31.2%, R2 of .462, R of .681 and a P Val of 2.04e^-17. These results confirm the results that bedrooms are predictable, however, with a moderately large error range.

# Install
pip install numpy | conda install numpy

pip install pandas | conda install pandas

pip install matplotlib | conda install matplotlib

pip install seaborn | conda install seaborn

pip install plotly

pip install cufflinks

pip install chart-studio

pip install -U scikit-learn

pip install scipy | conda install -c anaconda scipy

# Imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

init_notebook_mode(connected = True)

import chart_studio.plotly as py

cf.go_offline()

%matplotlib inline

import scipy.stats as st

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
