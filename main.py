# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers
#from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

marker_shapes = ['.', '^', '*']


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find the derivation of the
        # loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) *
                                            sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output)
                                                  * sigmoid_derivative(self.output), self.weights2.T) *
                                           sigmoid_derivative(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = Sequential()
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)
    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    print(nn.output)

    URL = \
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(URL, names=['sepal_length', 'sepal_width',
                                 'petal_length', 'petal_width', 'class'])
    print(df.info())
    print(df.describe())

    # Then, plot the scatterplot
    ax = plt.axes()
    for i, species in enumerate(df['class'].unique()):
        species_data = df[df['class'] == species]
        species_data.plot.scatter(x='sepal_length',
                                  y='sepal_width',
                                  marker=marker_shapes[i],
                                  s=100,
                                  title="Sepal Width vs Length by Species",
                                  label=species, figsize=(10, 7), ax=ax)

    df['petal_length'].plot.hist(title='Histogram of Petal Length')
    df.plot.box(title='Boxplot of Sepal Length & Width, and Petal Length & Width')

    df2 = pd.DataFrame({'Day': ['Monday', 'Tuesday', 'Wednesday',
                                'Thursday', 'Friday', 'Saturday',
                                'Sunday']})
    print(df2)
    print(pd.get_dummies(df2))

    # Randomly select 10 rows
    random_index = np.random.choice(df.index, replace=False, size=10)
    # Set the sepal_length values of these rows to be None
    df.loc[random_index, 'sepal_length'] = None

    df.sepal_length = df.sepal_length.fillna(df.sepal_length.mean())
    print(df.isnull().any())

    # Layer 1
    model.add(Dense(units=4, activation='sigmoid', input_dim=3))
    # Output Layer
    model.add(Dense(units=1, activation='sigmoid'))

    print(model.summary())

    sgd = optimizers.SGD(lr=1)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    np.random.seed(9)
    model.fit(X, y, epochs=1500, verbose=False)
    print(model.predict(X))
    plt.show()

