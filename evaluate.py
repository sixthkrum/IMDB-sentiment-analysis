import json
f = open('userDefinedParameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
model_name=param['model_name']
sequence_length = param['sequence_length']
#end

import matplotlib.pyplot as plt
import numpy as np

def visualizeTraining(hist):
    h=hist.history
    # Line Plot for plotting losses
    plt.plot(h['val_loss'], label='Validation_Loss')
    plt.plot(h['loss'],label='Training_Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Results/Loss_Line_Plot.png')

    plt.plot(h['val_accuracy'],label='Validation_Accuracy')
    plt.plot(h['accuracy'], label='Training_Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Results/Accuracy_Line_Plot.png')

def revert_Y_to_labels(yData):
    yLabels=[ 1 if rec > 0.5 else 0  for rec in yData ]
    return np.array(yLabels)

from model_architecture import model_framework
import os
def generateReport(X,Y):
    if model_name=="" or os.path.exists(model_name)==False:
        print("Kindly ensure that you train model before attempting to generate report")
        return
    model=model_framework()
    model.load_weights(model_name)
    scores = model.evaluate(X, Y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    predictedOutput=model.predict(X)
    from sklearn.metrics import confusion_matrix,classification_report
    predict_labels=revert_Y_to_labels(predictedOutput)
    actual_labels=Y

    classification_report_string = "Classification Report: \n" + str(classification_report(actual_labels,predict_labels,target_names=['Negative','Positive']))
    confusion_matrix_string = "Confusion Matrix: \n" + str(confusion_matrix(actual_labels,predict_labels))

    with open("Results/classification_report.txt", "w") as file:
        file.write(classification_report_string)
        file.write(confusion_matrix_string)

    print(classification_report_string)
    print(confusion_matrix_string)

    # print("Classification Report: \n", classification_report(Y_test, MNB.predict(X_test),target_names=['Negative','Positive']))
    # print("Confusion Matrix: \n", confusion_matrix(Y_test, MNB.predict(X_test)))

def saveModelArchitecture():
    from keras.utils.vis_utils import plot_model
    from model_architecture import model_framework
    model=model_framework()
    plot_model(model,to_file='Results/modelArchitecture_plot.png',show_layer_names=True)

import pickle
from preprocess import clean_data
from preprocess import sanityEmbeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences

def userTest():
    userInput = input("Please enter a review: \n")
    with open('vocab.pkl', 'rb') as pklFile:
        vocab = pickle.load(pklFile)

    with open('vectorizer.pkl', 'rb') as pklFile:
        vectorizer = pickle.load(pklFile)

    cleanedInput = clean_data(userInput)
    intInput = sanityEmbeddings([cleanedInput], vectorizer, vectorizer.build_tokenizer())
    intInput = pad_sequences(intInput , maxlen=sequence_length , padding='pre', value=0 )

    model=model_framework()
    model.load_weights(model_name)

    print("Output was predicted to be " + ("positive" if revert_Y_to_labels(model.predict(intInput)) == 1 else "negative"))
