# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 23:45:37 2021

@author: SARANSH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp

import warnings 
warnings.filterwarnings('ignore')

# Importing the Dataset
dataset = pd.read_csv("NuBank_Data.csv")

# DATA PREPROCESSING STEPS or DATA CLEANING

# Dropping the Rows for which 'Target Default' is NULL
dataset.dropna(subset=['target_default'], axis=0, inplace=True)

# Dropping the 'Target Fraud' columns since it has 96.6% values as NULL also the Irrelevant column
dataset.drop('target_fraud', axis=1, inplace=True)
#Dropping colums having only 1 unique value
dataset.drop(labels=['channel','external_data_provider_credit_checks_last_2_year'], axis=1, inplace=True)
# Trimming the Data by removing irrelevant columns
dataset.drop(labels=['facebook_profile','marketing_channel','job_name','lat_lon',
                    'user_agent','reason','zip','real_state','state','shipping_zip_code',
                    'shipping_state','profile_phone_number','ids','application_time_applied',
                    'email','application_time_in_funnel','external_data_provider_first_name','profile_tags'], axis=1, inplace=True)
# Dealing for the Outliers
dataset['reported_income']=dataset['reported_income'].replace(np.inf,np.nan)
dataset.loc[dataset['external_data_provider_email_seen_before']==-999, 'external_data_provider_email_seen_before'] = np.nan

dataset_num = dataset.select_dtypes(exclude='object').columns
dataset_cat = dataset.select_dtypes(include='object').columns

# Filling the Null values with ) since it is reasonable to believe that not every customer has taken a loan
dataset['last_amount_borrowed'].fillna(value=0,inplace=True)
dataset['last_borrowed_in_months'].fillna(value=0,inplace=True)
dataset['n_issues'].fillna(value=0, inplace=True)

#Filling the missing values in Numerical columns with 'Median'
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
dataset.loc[:,dataset_num] = imputer.fit_transform(dataset.loc[:,dataset_num])

#Filling the missing values in Categorical Columns with 'Mode/Most Frequent'
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataset.loc[:,dataset_cat] = imputer2.fit_transform(dataset.loc[:,dataset_cat])

bin_var = dataset.nunique()[dataset.nunique()==2].keys().tolist()
dataset_encoded = dataset.copy()
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
for col in bin_var:
    dataset_encoded[col] = le.fit_transform(dataset_encoded[col])
    
dataset_encoded = pd.get_dummies(dataset_encoded, columns=[x for x in dataset_cat if x not in bin_var])

#Dividing the Dataset into Independent and Dependent Variables
x = dataset_encoded.drop('target_default', axis =1)
y = dataset_encoded['target_default']

# Dividing into training and test dataset
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
x_train_rus, y_train_rus = rus.fit_resample(x_train,y_train)

# Defining a Function to calculate recall of the models
from sklearn.pipeline import make_pipeline
def val_model(x, y, clf):
    pipeline = make_pipeline(StandardScaler(),clf)
    scores = cross_val_score(pipeline, x, y, scoring='recall')
    
    return scores.mean()

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

xgb = XGBClassifier()
lgbm = LGBMClassifier()
ct = CatBoostClassifier()

model = []
recall = []

for clf in (xgb,lgbm,ct):
    model.append(clf.__class__.__name__)
    recall.append(val_model(x_train_rus,y_train_rus, clf))
    
recallParameter = pd.DataFrame(data=recall, index=model, columns=['Recall'])






xgb2 = XGBClassifier(n_estimators=50, max_depth=5, min_child_weight=6, gamma=1, learning_rate=0.0001)
lgbm2 = LGBMClassifier(max_depth=5, learning_rate=0.01, num_leaves=70,min_data_in_leaf=400)
ct2 = CatBoostClassifier(learning_rate=0.03, depth=6, l2_leaf_reg=5)

model2 = []
recall2 = []

for clf in (xgb2,lgbm2,ct2):
    model2.append(clf.__class__.__name__)
    recall2.append(val_model(x_train_rus,y_train_rus, clf))
    
recallParameter2 = pd.DataFrame(data=recall2, index=model2, columns=['Recall'])
