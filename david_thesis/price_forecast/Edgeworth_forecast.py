import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import json

# ML Stuff
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Activation
from keras.layers import Dense
from keras.metrics import MeanAbsoluteError

tf.config.set_visible_devices([], 'GPU')

# hyper params
input_length = 10
prediction_length = 7

# Enable LaTeX rendering
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (8, 3)

model_name = "edgeworth_test_model"


def preprocess_data(sequences, n_steps_in, n_steps_out, features):
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


def train_model(X_train, y_train):
    # define model
    model = Sequential()

    model.add(LSTM(32, activation='relu', input_shape=(input_length, 1)))
    # model.add(LSTM(50, activation='relu'))
    model.add(Dense(prediction_length))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[MeanAbsoluteError()])

    model.fit(X_train, y_train, epochs=100, batch_size=32, shuffle=False)
    model.save(f"models/{model_name}")
    return model


def main():
    # load data from file
    with open('price_history.json', 'r') as f:
        data = json.load(f)


    # extract all prices and waiting customers from episodes
    # prices is an matrix 30(ep) x 200(prices)
    prices_data = [[min(pair) for pair in data[i]] for i in range(len(data))]

    X, Y = preprocess_data(prices_data, n_steps_in=input_length, n_steps_out=prediction_length, features=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    # model = train_model(X_train, y_train)
    model = tf.keras.models.load_model(f"models/{model_name}")
    visualize(model, prices_data[-20:])


def predict(model, input_series):
    return model.predict(np.array(input_series).reshape((1, input_length, 1)), verbose=0)[0]


def visualize(model, data):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    plt.subplots_adjust(hspace=0.4, top=0.13)

    for i, ax in enumerate(axes.flat):
        episode = np.random.randint(0, len(data))
        ax.text(0.02, 0.08, f"episode {80 + episode}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
        forecast_location = np.random.randint(20, 180)

        time_series = data[episode]
        train = time_series[:forecast_location]
        test = time_series[forecast_location:]

        # Use the fitted model to make forecasts
        forecasts = predict(model, train[-input_length:])

        # Plot the actual data and the forecasts
        ax.step(np.arange(1, len(train) + 1), train, label="time series data")
        ax.step(np.arange(len(train) - input_length + 1, len(train) + 1), train[-input_length:], label="Input",
                color="orange")
        ax.step(np.arange(len(train), len(train) + len(test) + 1), train[-1:] + test, linestyle="dotted", color='gray')
        ax.step(np.arange(len(train), len(train) + len(forecasts) + 1), train[-1:] + list(forecasts), color="green",
                linestyle="--", linewidth=2, label="Forecasts")
        # ax.legend().remove()


        if i == 0:
            ax.set_ylabel('\\textbf{price}')
            ax.set_xlabel(r'\textbf{step}')

        ax.set_xlim([forecast_location - 30, forecast_location + 25])

        ax.set_xticklabels((np.array(ax.get_xticks().tolist())/2).astype(int))

    # Create a legend for all subplots
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*[lines_labels[0]])]

    axbox = axes.flat[1].get_position()
    fig.legend(lines, labels, loc='upper center', ncol=4, fancybox=True, shadow=True, bbox_to_anchor=[axbox.x0+0.5*axbox.width, axbox.y1+0.004], bbox_transform=fig.transFigure)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
