import numpy as np
import matplotlib as mtp
import pandas as pd

dataset = pd.read_csv("User_Data (1).csv")

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values
print(x)
print(y)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:,2:4])
print(x)
x[:,2:4]=imputer.transform(x[:,2:4])
print(x[1:15,2:4])

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

labelx= LabelEncoder()
x[:,1]=labelx.fit_transform(x[:,1])
ct= ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])], remainder="passthrough")
x=np.array(ct.fit_transform(x))
print(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_train)
print(x_test)














