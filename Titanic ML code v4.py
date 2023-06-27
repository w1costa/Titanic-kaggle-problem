import os, pylab, random, sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    # print(type(all_data))
    data = all_data.loc[:,['PassengerId', 'Survived','Pclass','Name','Sex','Age' ,'SibSp', 'Parch']]
    # data = data.dropna()
    average_age = data['Age'].mean()
    data['Age'].fillna(average_age, inplace=True)
    # data = data.head(25)
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
    # print(type(all_data))
    data = all_data.loc[:,['PassengerId','Pclass','Name','Sex','Age' ,'SibSp', 'Parch', 'Embarked']]
    average_age = data['Age'].mean()
    data['Age'].fillna(average_age, inplace=True)
    # data = data.dropna()
    # data = data.head(25)
    return data
                
def buildTitanicExamples(fileName):
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

#file = '/Users/wandacosta/Desktop/Python/Datasets/Titanic - ML problem from Kaggle/train.csv'
train_file = os.path.dirname(os.path.realpath(__file__)) + '/train.csv'
test_file = os.path.dirname(os.path.realpath(__file__)) + '/test.csv'
examples = buildTitanicExamples(train_file)    
# for i in range(10):
#     print(examples[i].getFeatures())

def accuracy(truePos, falsePos, trueNeg, falseNeg):
    numerator = truePos + trueNeg
    denominator = truePos + trueNeg + falsePos + falseNeg
    return numerator/denominator

def sensitivity(truePos, falseNeg):
    try:
        return truePos/(truePos + falseNeg)
    except ZeroDivisionError:
        return float('nan')
    
def specificity(trueNeg, falsePos):
    try:
        return trueNeg/(trueNeg + falsePos)
    except ZeroDivisionError:
        return float('nan')
    
def posPredVal(truePos, falsePos):
    try:
        return truePos/(truePos + falsePos)
    except ZeroDivisionError:
        return float('nan')
    
def negPredVal(trueNeg, falseNeg):
    try:
        return trueNeg/(trueNeg + falseNeg)
    except ZeroDivisionError:
        return float('nan')
       
def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True):
    """
    The function "getStats" calculates and prints the accuracy, sensitivity, specificity, and positive
    predictive value of a binary classification model based on the number of true positives, false
    positives, true negatives, and false negatives.
    
    :param truePos: The number of true positive cases in a binary classification problem, i.e., the
    number of cases where the model correctly predicted a positive outcome and the actual outcome was
    positive
    :param falsePos: false positives are the number of times the model predicted a positive outcome when
    the actual outcome was negative. In other words, it is the number of incorrect positive predictions
    made by the model
    :param trueNeg: The number of true negative cases in a binary classification problem. These are the
    cases where the model correctly predicts a negative outcome when the actual outcome is negative
    :param falseNeg: falseNeg refers to the number of negative instances that are incorrectly classified
    as positive by a model or test. In other words, it is the number of false negatives in a binary
    classification problem
    :param toPrint: A boolean parameter that determines whether the function should print the calculated
    statistics or not, defaults to True (optional)
    :return: A tuple containing the values of accuracy, sensitivity, specificity, and positive
    predictive value.
    """
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
    sens = sensitivity(truePos, falseNeg)
    spec = specificity(trueNeg, falsePos)
    ppv = posPredVal(truePos, falsePos)
    if toPrint:
        print('\nAccuracy =', round(accur, 3))
        print(' Sensitivity =', round(sens, 3))
        print(' Specificity =', round(spec, 3))
        print(' Pos. Pred. Val. =', round(ppv, 3))
    return (accur, sens, spec, ppv)

def split80_20(examples):
    """
    The function splits a given set of examples into a training set and a test set with a 80:20 ratio.
    
    :param examples: The parameter "examples" is a list of examples or instances that will be split into
    a training set and a test set
    :return: two lists: a training set and a test set. The training set contains 80% of the examples
    passed as input to the function, while the test set contains the remaining 20%.
    """
    sampleIndices = random.sample(range(len(examples)),len(examples)//5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    # print(len(trainingSet), 'training set examples and', len(testSet), 'test set examples\n')
    return trainingSet, testSet
    
def randomSplits(examples, method, numSplits, toPrint = True):
    """
    The function performs random splits on a dataset and calculates statistics based on the results of a
    given method.
    
    :param examples: a list of examples, where each example is a tuple of features and a label
    :param method: The method parameter is a function that takes in a training set and a test set as
    input and returns a tuple of four values: the number of true positives, false positives, true
    negatives, and false negatives
    :param numSplits: The number of times to split the data into training and testing sets and run the
    method
    :param toPrint: A boolean variable that determines whether or not the function should print the
    statistics of the splits, defaults to True (optional)
    :return: a tuple containing the average true positive rate, false positive rate, true negative rate,
    and false negative rate over the specified number of splits.
    """
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    # random.seed(0)
    for t in range(numSplits):
        trainingSet, testSet = split80_20(examples)
        results, w,x,y,z = method(trainingSet, testSet)
        # print(results)
        truePos += w
        falsePos += x
        trueNeg += y
        falseNeg += z
    getStats(truePos/numSplits, falsePos/numSplits,trueNeg/numSplits, falseNeg/numSplits, True)
    return truePos/numSplits, falsePos/numSplits, trueNeg/numSplits, falseNeg/numSplits

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
    
    # scaler = StandardScaler()
    # scaled_featureVecs = scaler.fit_transform(featureVecs)   
        
    model = LogisticRegression(max_iter = 1000)
    model.fit(featureVecs, labels)
    
    if toPrint:
        print('model.classes_ =', model.classes_)
        for i in range(len(model.coef_)):
            print('For label', model.classes_[1])
            for j in range(len(model.coef_[0])):
                print('   ', Passenger.featureNames[j], '=', model.coef_[0][j])
    return model

def applyModel(model, testSet, label, prob = 0.5):
    """
    The function applies a machine learning model to a test set and returns the number of true
    positives, false positives, true negatives, and false negatives based on a given label and
    probability threshold.
    
    :param model: The machine learning model that has been trained on a training set and is being used
    to make predictions on a test set
    :param testSet: testSet is a list of instances that we want to test our model on. Each instance in
    the list should have a set of features and a label
    :param label: The label parameter is the class label that we are interested in predicting. It is
    used to determine whether a prediction is a true positive, false positive, true negative, or false
    negative
    :param prob: The probability threshold for classifying a data point as positive or negative. If the
    predicted probability of a data point belonging to the positive class is greater than this
    threshold, it is classified as positive, otherwise it is classified as negative. The default value
    is 0.5
    :return: four values: truePos, falsePos, trueNeg, and falseNeg. These values represent the number of
    true positives, false positives, true negatives, and false negatives, respectively, for a given
    model, test set, label, and probability threshold.
    """
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    results = pd.DataFrame(columns = ('PassengerId', label, 'prob', 'prediction'))
    predictions = []
    for i in range(len(probs)):
        if probs[i][1] > prob:
            # print(testSet[i].getLabel())
            if testSet[i].getLabel() == 1:
                truePos += 1
            else:
                falsePos += 1
            prediction = 1
        else:
            if testSet[i].getLabel() != 1:
                trueNeg += 1
            else:
                falseNeg += 1
            prediction = 0
        predictions.append(prediction)
        results.loc[i] = [testSet[i].getPassengerId(), testSet[i].getLabel(), probs[i][1], prediction]
    # print('true positive:', truePos,'false positive:', falsePos, 'true negative:',trueNeg,'false negative:', falseNeg)
    # getStats(truePos, falsePos, trueNeg, falseNeg, True)
    # results.to_csv('/Users/wandacosta/Desktop/Python/Datasets/Titanic - ML problem from Kaggle/results.csv', index = False)
    return results, truePos, falsePos, trueNeg, falseNeg

def lr(trainingData, testData, prob = 0.5):
    """
    The function lr() builds a model using training data, applies the model to test data, and returns
    the results with a specified probability threshold.
    
    :param trainingData: This parameter is likely a dataset containing the training data for a machine
    learning model. It would typically include features (input variables) and labels (output variables)
    for each example in the dataset
    :param testData: The testData parameter is a dataset that contains the same columns as the
    trainingData parameter, except for the target variable 'Survived'. This dataset is used to evaluate
    the performance of the model built on the trainingData
    :param prob: The probability threshold for predicting survival. If the predicted probability of
    survival is greater than or equal to this threshold, the passenger is predicted to have survived. If
    the predicted probability is less than this threshold, the passenger is predicted to have not
    survived. The default value is 0.5, which means
    :return: The function `lr` returns the results of applying the logistic regression model built on
    the training data to the test data, with a specified probability threshold for classification. The
    results are the predicted survival outcomes for the test data.
    """
    model = buildModel(trainingData, False)
    # print(model)
    results, truePos, falsePos, trueNeg, falseNeg = applyModel(model, testData, 'Survived', prob)
    # print(results)
    return results, truePos, falsePos, trueNeg, falseNeg  

def submission_file(training_set, test_set, prob = 0.5):
    model = buildModel(training_set, False)
    # print(model)
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

training_set = buildTitanicExamples(train_file)
test_set = buildTestSet(test_file)
submission_file(training_set, test_set)

# print(examples)
# random.seed()
#numSplits = 20
# results = pd.DateFrame(columns = ('PassengerId', 'Survived', 'prob', 'prediction'))
#truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
#print('Average of', numSplits, '80/20 splits LR') # LR meaning Logistic Regression
#print('Average TruePos, FalsePos, TrueNeg, FalseNeg:', randomSplits(examples, lr, numSplits, True))
# getStats(truePos, falsePos, trueNeg, falseNeg, True)

def buildROC(trainingSet, testSet, title):
    """
    The function builds a ROC curve and calculates the area under the curve (AUROC) for a given training
    and test set, and plots the curve.
    
    :param trainingSet: The dataset used for training the model
    :param testSet: The test set is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on a separate training set. It is a set of data that the model
    has not seen before and is used to test the model's ability to generalize to new data. In this
    function,
    :param title: The title of the plot that will be generated by the function. It will include the
    AUROC (Area Under the Receiver Operating Characteristic) value
    :return: the area under the ROC curve (AUROC) as a float value. If the plot parameter is set to
    True, it will also display a plot of the ROC curve with the AUROC value as the title.
    """
    model = buildModel(trainingSet, False)
    xVals, yVals = [], []
    # results = pd.DataFrame(columns = ('PassengerId', 'Survived', 'prob', 'prediction'))
    p = 0.0
    while p <= 1.0:
        results, truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet,'Survived', p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    # print('xVals =', xVals, 'yVals =', yVals)
    auroc = sklearn.metrics.auc(xVals, yVals)
    
    plt.plot(xVals, yVals)
    plt.plot([0,1], [0,1])
    title = title + '\nAUROC = ' + str(round(auroc,3))
    plt.title(title)
    plt.xlabel('1 - specificity')
    plt.ylabel('Sensitivity')
    plt.show()

# auroc = 0.0
# trainingSet, testSet = split80_20(examples)
# lr(trainingSet, testSet, 0.5)
# buildROC(trainingSet, testSet, 'ROC for Predicting Survival, 1 Split')