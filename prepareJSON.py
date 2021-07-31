# will come from json file later
'''
vocabSize=5000
batch_size=1000
sequence_length=120
train=True
model_name=best_model.h5
num_of_epochs=15
'''
import json
user_defined_parameters={
    'vocabSize':5000,
    'batch_size':1000,
    'sequence_length':120,
    'train':1,
    'model_name':"best_model.h5",
    'num_of_epochs':30,
    'processData':0,
    'userTest':0
}

with open('userDefinedParameters.json', 'w') as outfile:
    json.dump(user_defined_parameters, outfile)
