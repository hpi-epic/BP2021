import random

from sklearn.model_selection import train_test_split
import tensorflow as tf

import numpy as np
import csv
import matplotlib.pyplot as plt

SPLIT_LENGTH = 80
tf.config.set_visible_devices([], 'GPU')


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_idx = i + n_steps
        # check if we are beyond the sequence
        if end_idx > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def load_price_data():
    with open('price_data.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        prices = data[0]  # assume there is only one line with prices
        return prices


def train_model(X_train, y_train):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, activation='relu', input_shape=(SPLIT_LENGTH, 1))))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss='mse')
    # fit model
    model.fit(X_train, y_train, epochs=100)
    model.save("prediction_model")
    return model


def predict(model, input_series, length_of_prediction=50):
    prediction = np.array(input_series)
    while len(prediction) < len(input_series) + length_of_prediction:
        prediction = np.append(prediction,
                               model.predict(prediction[-SPLIT_LENGTH:].reshape((1, SPLIT_LENGTH, 1)),
                                             verbose=0).item())
    return prediction


def main():
    # load pricing data
    prices = load_price_data()
    prices = [float(price) for price in prices]

    X, Y = split_sequence(prices, SPLIT_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    print(f"length of training data: {len(X_train)}")

    # model = train_model(X_train, y_train)
    prediction_length = 50

    model = tf.keras.models.load_model("prediction_model")

    for trial in range(25):
        plt.figure(figsize=(14, 5))

        # plot whole graph
        plt.plot(prices)

        split_idx = random.randint(0, 400)
        input_to_predictor = prices[split_idx:split_idx + SPLIT_LENGTH]

        plt.plot(range(split_idx, split_idx + SPLIT_LENGTH), input_to_predictor)

        predicted_series = predict(model, input_to_predictor, prediction_length)

        plt.plot(range(split_idx + SPLIT_LENGTH, split_idx + SPLIT_LENGTH + prediction_length), predicted_series[-prediction_length:])
        plt.show()



if __name__ == '__main__':
    main()
