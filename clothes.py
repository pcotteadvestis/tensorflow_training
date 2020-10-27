import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from typing import List, Union, Callable, Tuple
from transparentpath import TransparentPath as Path
import argparse

np.set_printoptions(linewidth=200)

model_directory = Path("models/")

# noinspection PyTypeChecker
parser = argparse.ArgumentParser(
    description="Tensorflow training", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--units", type=str, help="The number of nodes in each layers (Default : 10,128)", default=[128, 10],
)
parser.add_argument(
    "--activations",
    type=str,
    help="The activation functions names (Default : relu,softmax)",
    default=[tf.nn.relu, tf.nn.softmax],
)
parser.add_argument(
    "--first_layer", type=str, help="The first layer (Default : Flatten)", default=tf.keras.layers.Flatten,
)
parser.add_argument(
    "--shape", type=str, help="The number of nodes for the hidden layer (Default : 28,28)", default=[28, 28],
)
parser.add_argument(
    "--epochs", type=str, help="The number of epochs (Default : 10)", default=10,
)
parser.add_argument(
    "--rewrite", action="store_true", help="To recreate the output files",
)
parser.add_argument(
    "--kind", type=str, help="The name of the current iteration process", default="",
)
parser.add_argument("--force", action="store_true", help="To refit the model even if it is already saved")
args = parser.parse_args()


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get("accuracy") > 0.9:
            self.model.stop_training = True


def treat_args():
    units_ = args.units
    activations_ = args.activations
    first_layer_ = args.first_layer
    shape_ = args.shape
    epochs_ = args.epochs
    rewrite_ = args.rewrite
    kind_ = args.kind
    if isinstance(units_, str):
        units_ = [int(u) for u in units_.split(",")]
    if isinstance(activations_, str):
        activations_ = [getattr(tf.nn, act) for act in activations_.split(",")]
    if isinstance(first_layer_, str):
        first_layer_ = getattr(tf.keras.layers, first_layer_)
    if isinstance(shape_, str):
        shape_ = [int(u) for u in shape_.split(",")]
    if isinstance(epochs_, str):
        epochs_ = int(epochs_)
    return units_, activations_, first_layer_, shape_, epochs_, rewrite_, kind_


def create_model(
    units_: List[int],
    activations_: List[Callable],
    first_layer_: keras.layers.Layer = tf.keras.layers.Flatten,
    optimizer=tf.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=None,
    input_shape_=None,
    kind_="",
) -> Tuple[tf.keras.models.Model, Union[None, str], bool]:
    if input_shape_ is None:
        input_shape_ = [28, 28]
    if metrics is None:
        metrics = ["accuracy"]

    if len(units_) != len(activations_):
        if len(units_) < len(activations_) and len(units_) == 2:
            last_unit = units[-1]
            units_ = [units_[0]] * (len(activations_) - 1)
            units_.append(last_unit)
        elif len(activations_) < len(units_) and len(activations_) == 2:
            last_activation = activations_[-1]
            activations_ = [activations_[0]] * (len(units_) - 1)
            activations_.append(last_activation)
        else:
            raise ValueError("Units and activations must have the same length")

    layers = [first_layer_(input_shape=input_shape_)] + [
        tf.keras.layers.Dense(units_[i], activation=activations_[i]) for i in range(len(units_))
    ]

    act_names = (
        str([a.__name__ for a in activations_])
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .replace(",", "-")
        .replace("'", "")
    )
    name_ = (
        f"{len(layers)}layers_"
        f"{str(input_shape_).replace('[', '').replace(']', '').replace(' ', '').replace(',', 'x')}_"
        f"{str(units_).replace('[', '').replace(']', '').replace(' ', '').replace(',', '-')}_"
        f"{act_names}"
    )

    model_ = tf.keras.models.Sequential(layers)

    model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if (
        len(list((model_directory / name_).glob("/*"))) > 0
        and len(list((model_directory / name_).glob(f"/{kind_}*"))) != 0
        and not args.force
    ):
        return model_, name_, False
    return model_, name_, True


def get_value_kind(kind_):
    if kind_ == "nodes":
        return "_".join([str(u) for u in units])
    elif kind_ == "activations":
        return "_".join([a.__name__ for a in activations])
    elif kind_ == "layers":
        return len(units) - 1


def fit_model(model_):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Path("logs") / name)
    callbacks = MyCallback()
    hist = model_.fit(training_images, training_labels, epochs=epochs, callbacks=[tensorboard_callback, callbacks])
    (model_directory / name).mkdir()
    model_.save_weights((model_directory / name / f"model_{epochs}.ckpt").__fspath__())
    print(f"WEIGHTS SAVED IN {model_directory / name}")
    if kind != "":
        Path(model_directory / name / f"{kind}_{get_value_kind(kind)}.csv").write(
            data=pd.DataFrame.from_dict(hist.history)
        )


if __name__ == "__main__":

    units, activations, first_layer, shape, epochs, rewrite, kind = treat_args()

    model, name, do_fit = create_model(
        units_=units, activations_=activations, first_layer_=first_layer, input_shape_=shape, kind_=kind
    )

    mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    if do_fit:
        fit_model(model)
    else:
        model.load_weights((model_directory / name / f"model_{epochs}.ckpt").__fspath__())

    res = model.evaluate(test_images, test_labels)

    if kind != "":
        df = Path(model_directory / name / f"{kind}_{get_value_kind(kind)}.csv").read(index_col=0)
        df.loc["testing"] = res
        Path(model_directory / name / f"{kind}_{get_value_kind(kind)}.csv").write(df)
        print(df)

    # classifications = model.predict(test_images)
