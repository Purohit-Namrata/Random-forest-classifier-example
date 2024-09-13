import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Randomforestclassifierexample/500hits.csv",encoding='latin-1')
print(df.head())

df=df.drop(columns=['PLAYER','CS'])

X=df.iloc[:,0:13]
Y=df.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(n_estimators=1000,criterion='entropy',random_state=42)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)
print(y_pred)
print("Accuracy is ",accuracy_score(y_pred,y_test)*100,"%")