import json
f = open('userDefinedParameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
vocabSize=param['vocabSize']
sequence_length=param['sequence_length']
#end

# Working great
def classification_model_1(vocabSize,sequence_length,dropout_rate=0.2):
    from tensorflow.keras.activations import relu
    from tensorflow.keras.layers import Dense,Flatten
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import categorical_crossentropy
    from tensorflow.keras.utils import plot_model
    model = Sequential()
    model.add(Embedding(vocabSize ,32, input_length=sequence_length))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Stuck at 50%
def classification_model_LSTM(vocabSize,sequence_length,dropout_rate=0.2):
    from tensorflow.keras.activations import relu,softmax
    from tensorflow.keras.layers import Embedding,LSTM, Dropout, Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import categorical_crossentropy
    from tensorflow.keras.utils import plot_model

    # Parameters
    adam=Adam(lr=0.000003)

    model=Sequential()
    model.add(Embedding(vocabSize ,32, input_length=sequence_length))
    model.add(LSTM(32))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation=relu))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=adam ,
        loss='binary_crossentropy',
        metrics=[ 'accuracy' ]
    )
    print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def classification_model_new_LSTM(vocabSize=5000,sequence_length=120,dropout_rate=0.3):
    from tensorflow.keras.activations import relu
    from tensorflow.keras.layers import Embedding,LSTM, Dropout, Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model
    dropout_rate = 0.3
    from tensorflow.keras.activations import relu
    activation_func = relu
    SCHEMA = [
        Embedding( vocabSize , 10, input_length=sequence_length ),
        LSTM( 32 ) ,
        Dropout(dropout_rate),
        Dense( 32 , activation=activation_func ) ,
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ]
    model = Sequential(SCHEMA)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam() ,
        metrics=[ 'accuracy' ]
    )
    return model


def model_framework():
    return classification_model_new_LSTM(vocabSize=vocabSize, sequence_length=sequence_length,dropout_rate=0.3)
