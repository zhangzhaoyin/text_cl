from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
import os
import codecs
from sklearn import ensemble
import mysql.connector
from data_utils import cut,getLabels


class Models(object):

    def load_data(self):
        company_DATA = pd.read_csv(self.config.FLAGS.trainfile_dir, encoding="utf8")

        # company_DATA = company_DATA.drop_duplicates()
        print(company_DATA.shape)

        company_DATA = company_DATA[company_DATA.businessScope.notnull() & company_DATA.industry.notnull()]
        labels = pd.DataFrame.from_dict(getLabels())
        company_DATA = company_DATA.merge(labels, left_on="industry", right_on="secnduName")
        company_DATA = company_DATA[company_DATA.secInduCode.notnull()]
        company_DATA["businessScope_Cut"] = company_DATA["businessScope"].apply(cut)

        self.X = company_DATA["businessScope_Cut"]
        self.y = company_DATA["secInduCode"]

        return self.X, self.y

    def getTfidf(self):

        self.tv = TfidfVectorizer(sublinear_tf=True,
                             max_df=0.5,
                             max_features=5000)
        return self.tv


    def addRfmodel(self):

        self.clf = ensemble.RandomForestClassifier(criterion="entropy", random_state=1000)

        return self.clf


    def __init__(self,config):
        self.config = config
        self.load_data()
        self.getTfidf()
        self.addRfmodel()


    def trainTf(self):
        self.tv.fit(self.X)

        return self.tv

    def trianModel(self):
        print("模型训练...")
        print('*************************\nRF\n*************************')

        X = self.tv.transform(self.X)
        self.clf.fit(X, self.y)

        return self.clf
