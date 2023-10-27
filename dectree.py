import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
#%matplotlib inline

dataset=pd.read_csv("Social_Network_Ads.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

print(x_train)
print(x_test)

from sklearn.tree import DecisionTreeClassifier
classifiers=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifiers.fit(x_train, y_train)

print(classifiers.predict(sc.transform([[30,87000]])))

y_pred=classifiers.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01), np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.01))  
mtp.contourf(x1, x2, classifiers.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap = ListedColormap(['red','green']))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(['red', 'green'])(i), label = j)  
mtp.title('Decision Tree Algorithm (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()




#plt.figure(figsize=(15,10))    
#tree.plot_tree(classsifier, filled=True)
#print(tree.export_text(classifier))
