import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
data_set=pd.read_csv("/content/Salary_Data.csv")


x= data_set.iloc[:,:-1].values
y= data_set.iloc[:, -1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
st= StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)

y_pred= reg.predict(x_test)


from matplotlib.colors import ListedColormap

mtp.scatter(x_train, y_train, color='red')
mtp.plot(x_train,reg.predict(x_train), color='blue')
mtp.title('lin reg(Train set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()


mtp.scatter(x_test, y_test, color='red')
mtp.plot(x_train,reg.predict(x_train), color='blue')
mtp.title('lin reg(Test set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()
