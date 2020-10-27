import tensorflow as tf
import numpy as np
from tensorflow import keras

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=2000)


def house_model(y_new):
    return model.predict(y_new)[0]


if __name__ == "__main__":
    prediction = house_model([7.0])
    print(prediction)
