import json
f = open('userDefinedParameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
batch_size=param['batch_size']
model_name=param['model_name']
num_of_epochs=param['num_of_epochs']
#end

#Defining Our Deep Learning Model
from model_architecture import model_framework
from evaluate import visualizeTraining
def trainModel(xTrain,yTrain):
    model=model_framework()
    # Adding some checkpoints
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
    checkpoint=ModelCheckpoint(
        model_name,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto"
    )
    es_checkpoint=EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=2,
        verbose=0,
        mode="auto"
    )
    hist=model.fit( xTrain, yTrain, batch_size=batch_size, epochs=num_of_epochs,
        validation_split=0.20,callbacks=[checkpoint,es_checkpoint])
    visualizeTraining(hist)
