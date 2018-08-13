# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 08:54:13 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

df=pd.read_csv("lasvegas_tripadvisor.csv")


class LVR():
    
    
    def __init__(self):
        self.df = pd.read_csv("lasvegas_tripadvisor.csv")
        self.X=None
        self.y=[]
        self.n_estimators=500
        self.clf=None
        self.splitRatio=0.33
        self.trainX=[]
        self.trainY=[]
        self.testX=[]
        self.testY=[]
        self.validationAccuracies=[]
        self.kFold=5
        self.results=None
        self.models=[]
        self.finalAccuracy=0
        self.seasonDict={}
        
    def dropColumns(self):
        self.df.drop(['Nr. reviews','User continent','Review month','Review weekday','Helpful votes'], axis=1,inplace=True)
    def cleanseMemberYears(self):
        self.df['Member years']=self.df['Member years'].apply(lambda x: 0 if x<0 else x)
        
    def cleanseScores(self):
        self.df['Score']=self.df['Score'].apply(lambda x: 2 if x==1  else x)
        
    def binarizeCountry(self):
         self.df['Domestic']=self.df['User country'].apply(lambda x: 1 if x=='USA' else 0)
         self.df.drop(['User country'], axis=1,inplace=True)
         
    def formSeasonDict(self):
        
        self.seasonDict['Jan']='Winter'
        self.seasonDict['Feb']='Winter'
        self.seasonDict['Mar']='Spring'
        self.seasonDict['Apr']='Spring'
        self.seasonDict['May']='Spring'
        self.seasonDict['Jun']='Summer'
        self.seasonDict['Jul']='Summer'
        self.seasonDict['Aug']='Summer'
        self.seasonDict['Sep']='Fall'
        self.seasonDict['Oct']='Fall'
        self.seasonDict['Nov']='Fall'
        self.seasonDict['Dec']='Winter'
        
    def getSeasons(self):
        self.df['Season']=self.df['Period of stay'].apply(lambda x: x.split('-')[0]).apply(lambda x: self.seasonDict[x])
        self.df.drop(['Period of stay'], axis=1,inplace=True)
        
    def minMaxScale(self):
        columnList=['Nr. rooms','Member years']
        
        for column in columnList:
            listToBeNormalized=list(self.df[column])
            minValue=min(listToBeNormalized)
            maxValue=max(listToBeNormalized)
            listToBeNormalized=list(map(lambda x: (x-minValue)/(maxValue-minValue) , listToBeNormalized))
            self.df[column]=listToBeNormalized
            
    def normalizeColumns(self):
        columnList=['Nr. hotel reviews']
        
        for column in columnList:
            listToBeNormalized=list(self.df[column])
            mu=np.mean(listToBeNormalized)
            sigma=np.std(listToBeNormalized)
            listToBeNormalized=list(map(lambda x: (x-mu)/sigma , listToBeNormalized))
            self.df[column]=listToBeNormalized
            
    def encodeBinary(self):
        columnList=['Pool','Gym','Tennis court','Spa','Casino','Free internet']
        
        for column in columnList:
            le = preprocessing.LabelEncoder()
            self.df[column]=le.fit_transform(self.df[column].values.tolist())
            
    def labelBinarize(self):
        columnList=['Traveler type','Hotel name','Season']
        
        for column in columnList:
            lb = preprocessing.LabelBinarizer()
            lb.fit(self.df[column].tolist())
            transformedList=lb.transform(self.df[column].values.tolist())
            labels=list(lb.classes_)
            labels=list(map(lambda x: column+"_"+x , labels))
            for ind in range(len(labels)):
                self.df[labels[ind]]=transformedList[:,ind]
                
        self.df.drop(columnList, axis=1,inplace=True)
        
    def getXY(self):
        self.y=self.df['Score'].values.tolist()
        self.X=self.df.drop(['Score'], axis=1).values.tolist()
#        pca=PCA(n_components=5)
#        
#        print("pca components: ",len(self.X[0]))
#        self.X=pca.fit_transform(self.X)
#        print(pca.explained_variance_ratio_)
        
    def trainTestSplit(self):
        self.trainX, self.testX,self.trainY, self.testY = train_test_split(self.X, self.y, test_size=self.splitRatio, random_state=42)
    
    def trainAndValidate(self):    
        validationRatio=1/float(self.kFold)
            
        for validation in range(self.kFold):
               print("Validation number : ", validation)
               clf=RandomForestClassifier(n_estimators=self.n_estimators)
#               clf=AdaBoostClassifier(n_estimators=4500)        
               trainX, self.validateX,trainY, self.validateY = train_test_split(self.trainX, self.trainY, test_size=validationRatio)
               clf.fit(trainX,trainY)
               
               outcome=clf.predict(self.validateX)
               self.validationAccuracies.append(accuracy_score(outcome,self.validateY))
               self.models.append(clf)
        
        
        self.clf=self.models[self.validationAccuracies.index(max(self.validationAccuracies))]
        del self.models[:]
        print("Validation Accuracies: ")
        print(self.validationAccuracies)
        
        
    def test(self):
            self.results=self.clf.predict( self.testX)
            self.finalAccuracy=accuracy_score(self.results,self.testY) 
        
    def predictAndScore(self):
#        self.results=self.model.predict(self.testX)
        print("Accuracy Score: ", accuracy_score(self.results,self.testY ))
        print("Confusion Matrix: ")
        print( confusion_matrix(self.results,self.testY ))

    
        
    def printResults(self):
       
       for ii in range(len(self.results)):
           print(self.testY[ii],self.results[ii]) 
    
        
    def plot_coefficients(self):
        coef = self.clf.feature_importances_
 
         # create plot
        importances = pd.DataFrame({'feature':self.df.drop(['Score'], axis=1).columns.values,'importance':np.round(coef,3)})
        importances = importances.sort_values('importance',ascending=True).set_index('feature')
        print( importances)
        importances.plot.barh()     
        
        
        
        
if __name__ == '__main__':
    
    VR=LVR()
    VR.dropColumns()
    VR.cleanseMemberYears()
    VR.cleanseScores()
    VR.binarizeCountry()
    VR.formSeasonDict()
    VR.getSeasons()
    VR.minMaxScale()
    VR.normalizeColumns()
    VR.encodeBinary()
    VR.labelBinarize()
    VR.getXY()
    VR.trainTestSplit()
    VR.trainAndValidate()
    VR.test()
    VR.predictAndScore()
    VR.plot_coefficients()