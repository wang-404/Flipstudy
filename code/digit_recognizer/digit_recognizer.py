from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_train_file(filename):
    data = pd.read_csv(filename)
    data = data.to_numpy()
    Y = np.transpose(data[:,0])
    X = data[:,1:]
    return  X,Y

def read_test_file(filename):
    data = pd.read_csv(filename)
    data = data.to_numpy()
    return data

def split_train_dataset(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    return X_train,X_test,y_train,y_test

def updatetosubmission(y):
    pass

def train_and_evaluate(X,y):
    clf = RandomForestClassifier()
    X_train,X_test,y_train,y_test=split_train_dataset(X,y)
    clf.fit(X_train,y_train)
    y_result = clf.predict(X_test)
    accurary = np,sum(y_test == y_result)/y_test.shape[0]
    print("accuracy "+accurary)

def train_and_submit_test(trainX,trainY,testX):
    clf=RandomForestClassifier()
    clf.fit(trainX,trainY)
    result = clf.predict(testX)
    result = np.transpose(result)
    df = pd.DataFrame(result,columns=['label'])
    df.index=df.index+1
    df=df.to_csv("resulr.csv")

def main():
    train_X,train_y,=read_train_file("train.csv")
    test_X = read_test_file("test.csv")
    train_and_submit_test(train_X,train_y,test_X)

if __name__ == '__main__':
    main()



    