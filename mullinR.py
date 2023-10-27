import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
data_set=pd.read_csv("/content/50_comp.csv")


x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
labelencoder_x= LabelEncoder()  
x[:, 3]= labelencoder_x.fit_transform(x[:,3])  
onehotencoder= OneHotEncoder()   
x= onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.25, random_state=0)


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)

y_pred= reg.predict(x_test)

print('Train Score:', reg.score(x_train, y_train))
print('Train Score:', reg.score(x_test, y_test))