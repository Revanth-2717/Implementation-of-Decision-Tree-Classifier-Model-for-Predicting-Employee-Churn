# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Start the program
attach the given data file
now find the satisfaction level of employee data
find the accuracy and new predict value 5.end the program

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: REVANTH P
RegisterNumber: 212223040143

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![exp-8 out1](https://github.com/user-attachments/assets/aaa0a78a-74ac-4d84-8aff-91b42331102d)

Accuracy:
![accuracy](https://github.com/user-attachments/assets/6594aace-29c5-4e1e-a8f6-a3820cd873d0)

New predicted :
![new predic](https://github.com/user-attachments/assets/33ec94ba-553c-45ea-b04d-0b1766ede4c2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
