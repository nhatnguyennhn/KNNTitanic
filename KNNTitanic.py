import numpy  
import pandas 
from sklearn.neighbors import KNeighborsClassifier
DuLieuTrain=pandas.read_csv('train.csv')
DuLieuTest=pandas.read_csv('test.csv')
YTrain=DuLieuTrain.iloc[0:890,1]
XTrain=DuLieuTrain.iloc[0:890,[2,4,5]]
XTest=DuLieuTest.iloc[0:418,[1,3,4]]
x=[XTrain,XTest]
for i in x:
    i['Sex']=i['Sex'].map({'female':0,'male':1}).astype(int)
XTrain=(XTrain.fillna(0)) 
XTest=XTest.fillna(0)
knn=KNeighborsClassifier()
knn.fit(XTrain,YTrain)
duDoan=knn.predict(XTest)
ketQua=pandas.DataFrame(data=duDoan,index=DuLieuTest['PassengerId'],columns=['Survived'])
ketQua.to_csv('ketqua.csv',header=True)
