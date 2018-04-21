import csv
import random
import math
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import concurrent.futures
import operator
import string


class NaiveBaiesClassifier():
    def __init__(self):
        self.docCountOfClasses = {}
        self.tokenCountOfClasses = {}
        self.frequencies = {}
        self.uniqueTokensCount = 0

    def classify(self, text):
        classes = self.getClasses()

        tokens = self.tokenize(text)
        
        probsOfClasses = {}

        for className in classes:
            tokensProbs = [(self.getFrequency(token, className) + 1) for token in tokens]
            classTokenCount = self.getClassTokenCount(className)
            tokenSetProb = sum(math.log(tokenProb/(classTokenCount+self.uniqueTokensCount)) for tokenProb in tokensProbs)
            
            probsOfClasses[className] = tokenSetProb + math.log(self.getClassDocCount(className) /  self.getDocCount())
        
        return sorted(probsOfClasses.items(),key=operator.itemgetter(1),reverse=True)[0][0]
    
    
    def increaseClass(self, className, byAmount = 1):
        self.docCountOfClasses[className] = self.docCountOfClasses.get(className, 0) + 1

    def increaseToken(self, token, className, byAmount = 1):
        if not token in self.frequencies:
                self.frequencies[token] = {}
                self.uniqueTokensCount += 1

        self.frequencies[token][className] = self.frequencies[token].get(className, 0) + 1
    
    def increaseTokenCountOfClasses(self, token, className):
        self.tokenCountOfClasses[className] = self.tokenCountOfClasses.get(className, 0) + 1

    def getDocCount(self):
        return sum(self.docCountOfClasses.values())

    def getClasses(self):
        return self.docCountOfClasses.keys()

    def getClassDocCount(self, className):
        return self.docCountOfClasses.get(className, 0)
        
    def getClassTokenCount(self, className):
        return self.tokenCountOfClasses.get(className, 0)

    def getFrequency(self, token, className):
        if token in self.frequencies and className in self.frequencies[token]:
            foundToken = self.frequencies[token]
            return foundToken.get(className)
        else:
            return 0
            
    def train(self, text, className):
        self.increaseClass(className)

        tokens = self.tokenize(text)
        for token in tokens:
            self.increaseTokenCountOfClasses(token, className)
            self.increaseToken(token, className)
    
    def tokenize(self, text):
        stop = stopwords.words('english') + list(string.punctuation)
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return [stemmer.stem(i) for i in word_tokenize(text.lower()) if i not in stop]

			


filename = 'Reviews.csv'
df = pd.read_csv(filename, sep=',')
df.dropna(inplace=True)
mt = np.array(df[['Score', 'Text']])
splitRatio = 0.7
train, test = train_test_split(mt, test_size=(1-splitRatio))
	
classifier = NaiveBaiesClassifier()
for data in train:
    classifier.train(data[1], data[0])
    
guessed = 0
for data in test:
    g = classifier.classify(data[1])
    if g == data[0]:
        guessed += 1
    
print('model accuracy: {}%'.format(guessed * 100.0 / len(test)))

