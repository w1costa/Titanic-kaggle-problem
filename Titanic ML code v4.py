import os, pylab, random, sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

random.seed()

def minkowskiDist(v1, v2, p):
    """Assumes v1 and v2 are equal-length arrays of numbers
       Returns Minkowski distance of order p between v1 and v2"""
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p
    return dist**(1/p)

class Passenger(object):
    featureNames = ('C2', 'C3', 'age', 'male gender', 'SibSp', 'Parch')
    def __init__(self, pClass, age, gender, survived, name, passengerId, SibSp, Parch):
        self.name = name
        if pClass == 2:
            self.featureVec = [1, 0, age, gender, SibSp, Parch]
        elif pClass == 3:
            self.featureVec = [0, 1, age, gender, SibSp, Parch]
        else:
            self.featureVec = [0, 0, age, gender, SibSp, Parch]
        self.label = survived
        self.cabinClass = pClass
        self.passengerId = passengerId
        # if embark == 'C':
        #     self.featureVec.append(1)
        # elif embark == 'S':
        #     self.featureVec.append(2)
        # elif embark == 'Q':
        #     self.featureVec.append(3)
        # else:
        #     self.featureVec.append(4)
    def distance(self, other):
        return minkowskiDist(self.featureVec, other.featureVec, 2)
    def getClass(self):
        return self.cabinClass
    def getAge(self):
        return self.featureVec[3]
    def getGender(self):
        return self.featureVec[4]
    def getSibSp(self):
        return self.featureVec[5]
    def getParch(self):
        return self.featureVec[6]
    # def getEmbark(self):
    #     return self.featureVec[7]
    def getName(self):
        return self.name
    def getPassengerId(self):
        return self.passengerId
    def getFeatures(self):
        return self.featureVec[:]
    def getLabel(self):
        return self.label
        
def getTrainingData(fname):
    """
    The function reads data from a file containing information about passengers on the Titanic and
    returns a dictionary with their class, survival status, age, gender, and name.
    
    :param fname: The parameter `fname` is a string representing the file name or file path of the data
    file that contains information about the passengers on the Titanic
    :return: a DataFrame containing Titanic passenger data, including their passenger id, class, age,
    gender, survival status, and name.
    """
    all_data = pd.read_csv(fname)
    data = all_data.loc[:,['PassengerId', 'Survived','Pclass','Name','Sex','Age' ,'SibSp', 'Parch']]
    average_age = data['Age'].mean()
    data['Age'].fillna(average_age, inplace=True)
    return data

def getTestData(fname):
    """
    The function reads data from a file containing information about passengers on the Titanic and
    returns a dictionary with their class, age, gender, and name.
    
    :param fname: The parameter `fname` is a string representing the file name or file path of the data
    file that contains information about the passengers on the Titanic
    :return: a DataFrame containing Titanic passenger data, including their passenger id, class, age,
    gender, and name.
    """
    all_data = pd.read_csv(fname)
    data = all_data.loc[:,['PassengerId','Pclass','Name','Sex','Age' ,'SibSp', 'Parch', 'Embarked']]
    average_age = data['Age'].mean()
    data['Age'].fillna(average_age, inplace=True)
    return data
                
def buildTrainingSet(fileName):
    """
    This function builds a list of Passenger objects from Titanic data stored in a file.
    
    :param fileName: The name of the file containing the Titanic passenger data
    :return: a list of Passenger objects created from the data in the file specified by the fileName
    parameter.
    """
    data = getTrainingData(fileName)
    examples = []
    for _, row in data.iterrows():
        if row['Sex'] == 'male':
            p = Passenger(row['Pclass'], row['Age'], 1, row['Survived'], row['Name'], row['PassengerId'], row['SibSp'], row['Parch'])
        else:
            p = Passenger(row['Pclass'], row['Age'], 0, row['Survived'], row['Name'], row['PassengerId'],row['SibSp'], row['Parch'])
        examples.append(p)
    print('Finished processing', len(examples), 'passengers\n')    
    return examples

def buildTestSet(fileName):
    """
    This function builds a list of Passenger objects from Titanic data stored in a file.
    
    :param fileName: The name of the file containing the Titanic passenger data
    :return: a list of Passenger objects created from the data in the file specified by the fileName
    parameter.
    """
    data = getTestData(fileName)
    examples = []
    for _, row in data.iterrows():
        if row['Sex'] == 'male':
            p = Passenger(row['Pclass'], row['Age'], 1, 'NaN', row['Name'], row['PassengerId'], row['SibSp'], row['Parch'] )
        else:
            p = Passenger(row['Pclass'], row['Age'], 0, 'NaN', row['Name'], row['PassengerId'], row['SibSp'], row['Parch'])
        examples.append(p)
    print('Finished processing', len(examples), 'passengers\n')    
    return examples

def buildModel(examples, toPrint = True):
    """
    The function builds a logistic regression model using examples and prints the coefficients for each
    feature.
    
    :param examples: The examples parameter is a list of instances of a class that has a method called
    getFeatures() that returns a list of numerical features and a method called getLabel() that returns
    a label for the instance. These examples are used to train a logistic regression model
    :param toPrint: A boolean parameter that determines whether or not to print the model's information.
    If set to True, the function will print the model's classes and coefficients for each feature. If
    set to False, the function will not print anything and will only return the model, defaults to True
    (optional)
    :return: The function `buildModel` returns a trained logistic regression model.
    """
    
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
        
    model = LogisticRegression(max_iter = 1000)
    model.fit(featureVecs, labels)
    
    if toPrint:
        print('model.classes_ =', model.classes_)
        for i in range(len(model.coef_)):
            print('For label', model.classes_[1])
            for j in range(len(model.coef_[0])):
                print('   ', Passenger.featureNames[j], '=', model.coef_[0][j])
    return model

def submission_file(training_set, test_set, prob = 0.5):
    model = buildModel(training_set, False)
    testFeatureVecs = [e.getFeatures() for e in test_set]
    probs = model.predict_proba(testFeatureVecs)
    results = pd.DataFrame(columns = ('PassengerId', 'Survived'))
    predictions = []
    for i in range(len(probs)):
        if probs[i][1] > prob:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)
        results.loc[i] = [test_set[i].getPassengerId(), prediction]
    output_file = os.path.dirname(os.path.realpath(__file__)) + '/submission.csv'
    results.to_csv(output_file, index = False)
    return 

train_file = os.path.dirname(os.path.realpath(__file__)) + '/train.csv'
test_file = os.path.dirname(os.path.realpath(__file__)) + '/test.csv'    
training_set = buildTrainingSet(train_file)
test_set = buildTestSet(test_file)
submission_file(training_set, test_set)