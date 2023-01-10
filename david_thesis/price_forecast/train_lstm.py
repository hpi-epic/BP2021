import random
import sys

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Activation
from keras.layers import Dense
from keras.metrics import MeanAbsoluteError

import numpy as np
import csv
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

# hyper params
input_length = 10
prediction_length = 7


def preprocess_multistep_lstm(sequences, n_steps_in, n_steps_out, features):
    X, y = list(), list()

    for sequence in sequences:
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], features))

    return X, y


def load_price_data():
    with open('price_data.csv', newline='') as f:
        return [list(map(float, row)) for row in csv.reader(f)]


def train_model(X_train, y_train):
    # define model
    model = Sequential()

    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(input_length, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(prediction_length))
    model.add(Activation('linear'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[MeanAbsoluteError()])

    # fit model
    model.fit(X_train, y_train, epochs=100, batch_size=32, shuffle=False)
    model.save("models/prediction_model_7")
    return model


def predict(model, input_series):
    return model.predict(np.array(input_series).reshape((1, input_length, 1)), verbose=0)


def main():
    # load pricing data
    prices = load_price_data()

    X, Y = preprocess_multistep_lstm(prices, n_steps_in=input_length, n_steps_out=prediction_length, features=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    # print(f"length of training data: {len(X_train)}")

    model = train_model(X_train, y_train)

    # model = tf.keras.models.load_model("prediction_model_shorter")

    for trial in range(15):
        plt.figure(figsize=(14, 5))

        test_idx = random.randint(0, X_test.shape[0])

        # actual
        plt.plot(np.concatenate([X_test[test_idx].squeeze(), y_test[test_idx]]))
        plt.xticks(np.arange(0, input_length + prediction_length + 1))

        predicted_series = predict(model, X_test[test_idx])
        plt.plot(range(input_length, input_length + prediction_length), predicted_series.squeeze())
        plt.show()


if __name__ == '__main__':
    main()
