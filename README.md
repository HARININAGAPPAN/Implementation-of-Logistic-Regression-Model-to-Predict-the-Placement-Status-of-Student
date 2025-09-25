# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
# AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

# Equipments Required:

Hardware – PCs

Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

Import the required packages and print the present data

Print the placement data and salary data.

Find the null and duplicate values.

Using logistic regression find the predicted values of accuracy , confusion matrices.

# Program:
```
/*
Developed by: Harini N
RegisterNumber: 212223040057

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```
# Output:
# TOP 5 ELEMENTS
<img width="918" height="162" alt="image" src="https://github.com/user-attachments/assets/06d80eff-3e20-459b-83af-3b0af1a7defc" />

<img width="912" height="420" alt="image" src="https://github.com/user-attachments/assets/52aa8ff9-db92-4dd3-b371-ac02cc0feada" />

<img width="910" height="190" alt="image" src="https://github.com/user-attachments/assets/345d7792-6a06-4779-831a-4b76b92da4ec" />

DATA DUPLICATE

<img width="65" height="40" alt="image" src="https://github.com/user-attachments/assets/f2292b18-8102-44e1-988d-3b49db6d22c7" />

PRINT DATA

<img width="902" height="463" alt="image" src="https://github.com/user-attachments/assets/6fa5da74-2cc5-4496-b16d-671b3589e46f" />

DATA STATUS

<img width="651" height="289" alt="image" src="https://github.com/user-attachments/assets/c682a499-6adb-4f25-a80a-ee073554fe71" />

Y_PREDDICTION ARRAY

<img width="723" height="57" alt="image" src="https://github.com/user-attachments/assets/97dbab2e-7057-4291-b7fd-ad318641b121" />

CONFUSION ARRAY

<img width="910" height="497" alt="image" src="https://github.com/user-attachments/assets/8913a2b1-a031-4ede-9e41-f2df86a0a547" />

ACCURACY VALUE

<img width="199" height="47" alt="image" src="https://github.com/user-attachments/assets/79250063-0790-4c4c-bc41-907d812ce867" />

CLASSIFICATION REPORT

<img width="564" height="184" alt="image" src="https://github.com/user-attachments/assets/c96ec950-01a3-4952-b379-88adcea8da32" />

PREDICTION

<img width="259" height="31" alt="image" src="https://github.com/user-attachments/assets/73e6a8d8-e480-4e90-8a50-9d0a4a626e92" />

# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
