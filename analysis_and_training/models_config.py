import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

#DATA_PATH = "../13/data_10.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)
    plt.figure(figsize=(16,8))

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size, DATA_PATH):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis] # 4d array -> (num_sample, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_cnn_model(input_shape):
    """Generates CNN model
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def build_rnn_model(input_shape):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()
    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def build_ann_model(input_shape):
    model = keras.Sequential()

    # input layer
    model.add(keras.layers.Flatten(input_shape=input_shape)),

    # 1st dense layer
    model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))),
    model.add(keras.layers.Dropout(0.3)),

    # 2nd dense layer
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))),
    model.add(keras.layers.Dropout(0.3)),

    # 3rd dense layer
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))),
    model.add(keras.layers.Dropout(0.3)),

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model





def compule_model(lr, loss, metrics, model):
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))
    
def model_loss_acc(model, X_train, y_train, X_test, y_test, X_validation, y_validation):
    score_train = model.evaluate(X_train, y_train,  verbose=0)
    score_test  = model.evaluate(X_test, y_test,  verbose=0)
    score_val  = model.evaluate(X_validation, y_validation,  verbose=0)
    
    train_acc = round(score_train[1]*100, 2)
    test_acc = round(score_test[1]*100, 2)
    val_acc = round(score_val[1]*100, 2)
    
    train_loss = round(score_train[0]*100, 2)
    test_loss = round(score_test[0]*100, 2)
    val_loss = round(score_val[0]*100, 2)
    st_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    gen = 100 - test_acc
    return pd.DataFrame({'Accuracy':[train_acc, val_acc, test_acc],  
                         'Loss':[train_loss, val_loss, test_loss]}, 
                        index=['Training', 'Validation', 'Testing']).T

# if __name__ == "__main__":

#     # get train, validation, test splits
#     X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

#     # create network
#     input_shape = (X_train.shape[1], X_train.shape[2], 1)
#     model = build_model(input_shape)

#     # compile model
#     optimiser = keras.optimizers.Adam(learning_rate=0.0001)
#     model.compile(optimizer=optimiser,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])

#     model.summary()

#     # train model
#     history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

#     # plot accuracy/error for training and validation
#     plot_history(history)

#     # evaluate model on test set
#     test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#     print('\nTest accuracy:', test_acc)

#     # pick a sample to predict from the test set
#     X_to_predict = X_test[100]
#     y_to_predict = y_test[100]

#     # predict sample
#     predict(model, X_to_predict, y_to_predict)