# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/4 14:42 
# @Contact: 940942500@qq.com
import pandas as pd
from data_utils import cut,getLabels
from sklearn.model_selection import train_test_split
from modelTrain import Models
import pandas as pd

class Config(object):

    trainFilename = "D:/company_DATA.csv"
    testFile = "D:/zzy@creditease.cn/Company.csv"






if __name__ == '__main__':
    config = Config()
    model = Models(config)
    tf = model.trainTf()
    clf = model.trianModel()

    testData = pd.read_csv(config.testFile)
    testData["cut"] = testData["scapeOfBesiness"].apply(cut)
    X_test = tf.transform(testData["cut"])
    y_test = clf.predict(X_test)
    testData["secInduCode"] = pd.DataFrame(y_test)
    labels = pd.DataFrame.from_dict(getLabels())
    testData = testData.merge(labels, on="secInduCode",how="left")

    testData.to_csv("D:/zzy@creditease.cn/Company22.csv",index= None,encoding="utf8")

    print(testData[["companyName","secnduName"]].head())

    # print("训练集和测试集切分...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)
    # X_train_L = tfidf.transform(X_train)
    # X_test_L = tfidf.transform(X_test)