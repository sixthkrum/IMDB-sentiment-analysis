import json
f = open('userDefinedParameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
train=param['train']==1
processData=param['processData']==1
user_test = param['userTest'] == 1

from evaluate import userTest
if user_test:
    userTest()
    exit()

# Loading Datset
from preprocess import load_data_keras_preproccesed,load_data_self_preprocess
(xTrain,yTrain),(xTest,yTest)= load_data_self_preprocess(processData=processData)
# (xTrain,yTrain),(xTest,yTest)= load_data_keras_preproccesed(processData=processData)

# Save Model Diagram (not working as of now)
from evaluate import saveModelArchitecture
saveModelArchitecture()


# Loading Code to Train Model
from modelTraining import trainModel
if train==True:
    trainModel(xTrain,yTrain)
# Loading Code to Evaluate the results on test data
from evaluate import generateReport
generateReport(xTest,yTest)
