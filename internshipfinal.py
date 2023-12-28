

import pandas as pd
import numpy as np
import pickle

data=pd.read_csv('HR_comma_sep.csv')

data.drop_duplicates(inplace=True)

data.rename(columns = {'time_spend_company':'years_spend_comp'}, inplace = True)

data=data.drop(['sales','last_evaluation'],axis=1)

def remove_outlier(data,columns):
  for column in columns:
    q1=data[column].quantile(0.25)
    q3=data[column].quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    data_fi=data[(data[column]>=lower)&(data[column]<=upper)]
  return data_fi

columns=['promotion_last_5years','left','Work_accident','years_spend_comp']
data=remove_outlier(data,columns)

from sklearn.preprocessing import OrdinalEncoder

# Assuming 'data' is your DataFrame
oe = OrdinalEncoder()
data.info()
# Reshape the 'salary' column
salary_column_reshaped = data['salary'].values.reshape(-1, 1)

# Fit and transform the reshaped column
data['salary'] = oe.fit_transform(salary_column_reshaped)

x =data.drop('left',axis=1)
y = data['left']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier


rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

pickle.dump(rf_classifier,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))