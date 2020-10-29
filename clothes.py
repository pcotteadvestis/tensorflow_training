import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import numpy as np
from transparentpath import TransparentPath as Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

np.set_printoptions(linewidth=200)

model_directory = Path("models/")
log_directory = Path("logs/")


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get("accuracy") > 0.99:
            self.model.stop_training = True


def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {"compressedHistograms": 10, "images": 0, "scalars": 100, "histograms": 1}

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    # print(event_acc.Tags())

    epoch_loss = event_acc.Scalars("epoch_loss")
    epoch_accuracy = event_acc.Scalars("epoch_accuracy")

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = epoch_loss[i][2]
        y[i, 1] = epoch_accuracy[i][2]

    plt.plot(x, y[:, 0], label="Loss")
    plt.plot(x, y[:, 1], label="Accuracy")

    plt.xlabel("Steps")
    plt.ylabel("Accuracy and Loss")
    plt.title("Training Progress")
    plt.legend(loc="upper right", frameon=True)
    plt.show()


class MyModel:
    def __init__(self, args_, *args, **kwargs):
        self.units = []
        self.activations = []
        self.first_layer = None
        self.shape = None
        self.epochs = None
        self.rewrite = False
        self.convolute = None
        self.input_path = None
        self.directories = []
        self.rgb = False
        self.loss = ""
        self.optimizer = None
        self.metrics = ["accuracy"]
        self.optimizer_kwargs = {}
        self.training_images, self.training_labels, self.test_images, self.test_labels = None, None, None, None
        self.train_generator = None
        self.mnist = None
        self.force = False
        self.do_fit = True
        self.weights_path = None
        self.verbose = 0
        self.previous_log_names = None
        self.logfile = None

        self.treat_args(args_)

        self.layers = None
        self.name = ""

        self.load_data()
        self.create_model()

        self.model = tf.keras.models.Sequential(self.layers, *args, **kwargs)
        # noinspection PyCallingNonCallable

        self.model.compile(optimizer=self.optimizer(**self.optimizer_kwargs), loss=self.loss, metrics=self.metrics)

        print("\nARGUMENTS:")
        print(f"- Units: {self.units}")
        print(f"- Activations: {self.activations}")
        print(f"- First Layer: {self.first_layer}")
        print(f"- Shape: {self.shape}")
        print(f"- Epochs: {self.epochs}")
        print(f"- Rewrite: {self.rewrite}")
        print(f"- Convolute: {self.convolute}")
        print(f"- Input path: {self.input_path}")
        print(f"- RGB: {self.rgb}")
        print(f"- loss function: {self.loss}")
        print(f"- Optimizer: {self.optimizer}")
        print(f"- Optimizer kwargs: {self.optimizer_kwargs}")

        if self.verbose > 1:
            self.preview_input()

        if self.verbose > 0:
            self.model.summary()

    def load_data(self):
        if self.mnist is not None:
            (self.training_images, self.training_labels), (self.test_images, self.test_labels) = self.mnist.load_data()

            if self.shape is None:
                self.shape = self.training_images[0].shape
            for image in self.training_images:
                if image.shape != self.shape:
                    raise ValueError(
                        f"An image did not have the correct shape in trainign images : {image.shape} vs"
                        f" {self.shape}"
                    )
            for image in self.test_images:
                if image.shape != self.shape:
                    raise ValueError(f"An image did not have the correct shape : {image.shape} vs {self.shape}")

            if self.convolute is not None:
                self.test_images = self.test_images.reshape(10000, 28, 28, 1)
                self.training_images = self.training_images.reshape(60000, 28, 28, 1)

            self.training_images = self.training_images / 255.0
            self.test_images = self.test_images / 255.0
        else:
            train_datagen = ImageDataGenerator(rescale=1 / 255)
            self.train_generator = train_datagen.flow_from_directory(
                self.input_path,
                target_size=self.shape,
                batch_size=128,
                class_mode="binary" if "binary" in self.loss else "categorical",
            )

    def treat_args(self, args_):
        self.units = args_.units
        self.activations = args_.activations
        self.first_layer = args_.first_layer
        self.shape = args_.shape
        self.epochs = args_.epochs
        self.rewrite = args_.rewrite
        self.convolute = args_.convolute
        self.input_path = args_.input
        self.force = args_.force
        self.verbose = args_.verbose

        if "tf.keras.datasets" in self.input_path:
            self.mnist = getattr(tf.keras.datasets, self.input_path.split("tf.keras.datasets.")[1])
        else:
            self.input_path = Path(self.input_path)
            self.directories = list(self.input_path.glob("/*"))

        self.rgb = args_.rgb
        self.loss = args_.loss
        self.optimizer = getattr(tf.optimizers, args_.optimizer)
        if args_.optimizer_kwargs == "":
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = {
                keyval.split(":")[0]: keyval.split(":")[1] for keyval in args_.optimizer_kwargs.split(",")
            }
        for arg in self.optimizer_kwargs:
            if "%i" in self.optimizer_kwargs[arg]:
                # noinspection PyUnresolvedReferences
                self.optimizer_kwargs[arg] = int(self.optimizer_kwargs[arg].replace("%i", ""))
            if "%f" in self.optimizer_kwargs[arg]:
                # noinspection PyUnresolvedReferences
                self.optimizer_kwargs[arg] = float(self.optimizer_kwargs[arg].replace("%f", ""))
            if "%b" in self.optimizer_kwargs[arg]:
                # noinspection PyUnresolvedReferences
                self.optimizer_kwargs[arg] = bool(self.optimizer_kwargs[arg].replace("%b", ""))
        if isinstance(self.units, str):
            self.units = [int(u) for u in self.units.split(",")]
        if isinstance(self.activations, str):
            self.activations = [getattr(tf.nn, act) for act in self.activations.split(",")]
        if isinstance(self.first_layer, str):
            self.first_layer = getattr(tf.keras.layers, self.first_layer)
        if isinstance(self.shape, str):
            self.shape = tuple([int(u) for u in self.shape.split(",")])
        if isinstance(self.epochs, str):
            self.epochs = int(self.epochs)
        if isinstance(self.convolute, str):
            self.convolute = [int(c) for c in self.convolute.split(",")]

        # for kind in ["nodes", "layers", "activations"]:
        #     self.outname = f"{self.outname}{kind}__{self.get_value_kind(kind)}__"
        # self.outname = self.outname[:-2]

    def create_model(self):
        if self.metrics is None:
            self.metrics = ["accuracy"]

        if len(self.units) != len(self.activations):
            if len(self.units) < len(self.activations) and len(self.units) == 2:
                last_unit = self.units[-1]
                self.units = [self.units[0]] * (len(self.activations) - 1)
                self.units.append(last_unit)
            elif len(self.activations) < len(self.units) and len(self.activations) == 2:
                last_activation = self.activations[-1]
                self.activations = [self.activations[0]] * (len(self.units) - 1)
                self.activations.append(last_activation)
            else:
                raise ValueError("Units and activations must have the same length")

        if self.convolute is not None:
            self.shape = list(self.shape)
            # noinspection PyUnresolvedReferences
            self.shape.append(3 if self.rgb else 1)
            self.shape = tuple(self.shape)
            self.layers = []
            for i in range(len(self.convolute)):
                if i == 0:
                    self.layers.append(
                        tf.keras.layers.Conv2D(self.convolute[i], (3, 3), activation="relu", input_shape=self.shape)
                    )
                else:
                    self.layers.append(tf.keras.layers.Conv2D(self.convolute[i], (3, 3), activation="relu"))
                self.layers.append(tf.keras.layers.MaxPooling2D(2, 2))
            self.layers.append(self.first_layer(input_shape=[self.shape[0] - 2, self.shape[1] - 2]))
            self.layers += [
                tf.keras.layers.Dense(self.units[i], activation=self.activations[i]) for i in range(len(self.units))
            ]
            convoluted = (
                f"_convoluted_"
                f"{str(self.convolute).replace('[', '').replace(']', '').replace(' ', '').replace(',', '-')}_"
            )
        else:
            convoluted = ""
            self.layers = [self.first_layer(input_shape=self.shape)] + [
                tf.keras.layers.Dense(self.units[i], activation=self.activations[i]) for i in range(len(self.units))
            ]

        act_names = (
            str([a.__name__ for a in self.activations])
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .replace(",", "-")
            .replace("'", "")
        )

        self.name = (
            f"{self.input_path.parent if isinstance(self.input_path, Path) else self.input_path.split('.')[-1]}_"
            f"{len(self.layers) - 2}layers_"
            f"{str(self.shape).replace('(', '').replace(')', '').replace(' ', '').replace(',', 'x')}_"
            f"{str(self.units).replace('[', '').replace(']', '').replace(' ', '').replace(',', '-')}_"
            f"{act_names}{convoluted}"
        )

        self.weights_path = model_directory / self.name / f"model_{self.epochs}.ckpt"

        if len(list(self.weights_path.parent.glob("/*"))) > 0 and not self.force:
            self.do_fit = False
        else:
            self.do_fit = True

    def get_value_kind(self, kind: str = None):
        if kind == "nodes":
            return "-".join([str(u) for u in self.units[:-1]])
        elif kind == "activations":
            return "-".join([a.__name__ for a in self.activations])
        elif kind == "layers":
            return len(self.units) - 1

    def fit_model(self):
        if self.do_fit:

            self.previous_log_names = log_directory / self.name / "train"
            if self.previous_log_names.is_dir():
                self.previous_log_names = list(self.previous_log_names.glob("/*"))
            else:
                self.previous_log_names = None

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory / self.name)
            callbacks = MyCallback()
            if self.mnist is not None:
                self.model.fit(
                    self.training_images,
                    self.training_labels,
                    epochs=self.epochs,
                    callbacks=[tensorboard_callback, callbacks],
                )
            else:
                self.model.fit(
                    self.train_generator,
                    steps_per_epoch=8,
                    epochs=self.epochs,
                    callbacks=[tensorboard_callback, callbacks],
                    verbose=1,
                )

            current_log_names = list((log_directory / self.name / "train").glob("/*"))
            for name in current_log_names:
                if name not in self.previous_log_names:
                    self.logfile = name

            self.weights_path.parent.mkdir()
            self.model.save_weights(self.weights_path.__fspath__())
            print(f"\nWEIGHTS SAVED IN {self.weights_path.parent}\n")
        else:
            self.model.load_weights(self.weights_path.__fspath__())

    def evaluate(self):
        result = self.model.evaluate(self.test_images, self.test_labels)
        return result

    def predict(self):
        return self.model.predict(self.test_images)

    def preview_input(self):
        if self.mnist is None:
            nrows = 4
            ncols = 2 * len(self.directories)
            gs1 = gridspec.GridSpec(nrows, ncols)
            gs1.update(wspace=0.025, hspace=0.05)

            all_picutres = []
            for directory in self.directories:
                all_picutres.append(list(directory.glob("/*")))
                random.shuffle(all_picutres[-1])

            pictures = [picture for directory in all_picutres for picture in directory[0:8]]
        else:
            nrows = 4
            ncols = 4
            gs1 = gridspec.GridSpec(nrows, ncols)
            gs1.update(wspace=0.025, hspace=0.05)
            all_picutres = self.training_images[:]
            random.shuffle(all_picutres)
            pictures = [picture for picture in all_picutres[0:8]]

        for i, img_path in enumerate(pictures):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(gs1[i])
            sp.axis("Off")  # Don't show axes (or gridlines)
            if isinstance(img_path, Path):
                img = mpimg.imread(img_path.open("br"))
            else:
                img = img_path
            plt.imshow(img)

        plt.show()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Tensorflow training", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--units", type=str, help="The number of nodes in each layers (Default : 128,10)", default=[128, 10],
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
        "--shape", type=str, help="The number of nodes for the hidden layer (Default : 28,28)", default=(28, 28),
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
    parser.add_argument(
        "--convolute", type=str, help="The number of nodes in each convolution layers (Default : None)", default=None,
    )
    parser.add_argument(
        "--input", type=str, help="The path to input data", default="tf.keras.datasets.fashion_mnist",
    )
    parser.add_argument(
        "--rgb", action="store_true", help="If you use colored images",
    )
    parser.add_argument(
        "--loss", type=str, help="Loss function to use", default="sparse_categorical_crossentropy",
    )
    parser.add_argument(
        "--optimizer", type=str, help="The optimizer to use", default="Adam",
    )
    parser.add_argument(
        "--optimizer_kwargs", type=str, help="Arguments to pass to the optimizer", default="",
    )
    parser.add_argument(
        "--verbose", type=int, help="verbosity level", default=0,
    )
    parser.add_argument("--force", action="store_true", help="To refit the model even if it is already saved")
    args = parser.parse_args()

    model = MyModel(args)

    model.fit_model()

    res = model.evaluate()

    classifications = model.predict()

    plot_tensorflow_log(model.logfile.__fspath__())
