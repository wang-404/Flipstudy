from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
Boston = load_boston()
X = Boston.data   #特征
y = Boston.target  #房价
#将特征和标签  划分为训练集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=888)
lin_req = LinearRegression()
lin_req.fit(X_train,y_train)
lin_req.score(X_test,y_test)
print(lin_req.predict(X_test))#预测结果
print()
deviation = lin_req.predict(X_test)-y_test#偏差
print(deviation)
RMSE = np.sum(np.sqrt(deviation*deviation))/102
print(RMSE)
result = {'prediction':lin_req.predict(X_test)}
result_file = pd.DataFrame(result)
print(result_file)
result_file.to_csv("result.csv")