from typing import Tuple, Dict

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# print(tensorflow.__version__)

def set_up(layer_units, n_inputs, first_layer, activation):
    model = keras.Sequential()
    if activation == "relu":
        model.add(layers.Dense(first_layer, activation="relu", use_bias=False, input_dim=n_inputs))
        for unit in layer_units:
            model.add(layers.Dense(unit, activation="relu"))
    if activation == "tanh":
        model.add(layers.Dense(first_layer, activation="tanh", use_bias=False, input_dim=n_inputs))
        for unit in layer_units:
            model.add(layers.Dense(unit, activation="tanh"))

    return model


def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models. Dimension, 8 arguments for this dataset.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """

    deep_units = [3, 9, 15, 18, 20, 24, 33, 36, 43, 48]
    model_deep = set_up(deep_units, n_inputs, first_layer=4, activation="relu")
    model_deep.add(layers.Dense(n_outputs, activation="linear", use_bias=False))
    model_deep.compile(optimizer='rmsprop', loss='mse')

    wide_units = [65, 100]
    model_wide = set_up(wide_units, n_inputs, first_layer=4, activation="relu")
    model_wide.add(layers.Dense(n_outputs, activation="linear", use_bias=False))
    model_wide.compile(optimizer='rmsprop', loss='mse')

    return (model_deep, model_wide)


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """

    units = [200, 300]
    model_relu = set_up(units, n_inputs, first_layer=100, activation="relu")
    model_relu.add(layers.Dense(n_outputs, activation="sigmoid"))
    model_relu.compile(optimizer='rmsprop', loss='hinge')

    model_tanh = set_up(units, n_inputs, first_layer=100, activation="tanh")
    model_tanh.add(layers.Dense(n_outputs, activation="sigmoid"))
    model_tanh.compile(optimizer='rmsprop', loss='hinge')

    return (model_relu, model_tanh)


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    model_dropout = keras.Sequential()
    model_dropout.add(layers.Dense(100, activation="relu", input_dim=n_inputs))
    model_dropout.add(layers.Dense(200, activation="relu"))
    model_dropout.add(layers.Dropout(0.14))
    model_dropout.add(layers.Dense(300, activation="relu"))
    model_dropout.add(layers.Dense(n_outputs, activation="softmax"))
    model_dropout.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    units = [200, 300]
    model_no_dropout = keras.Sequential()
    model_no_dropout.add(layers.Dense(100, activation="relu", input_dim=n_inputs))

    for u in units:
        model_no_dropout.add(layers.Dense(u, activation="relu"))
    model_no_dropout.add(layers.Dense(n_outputs, activation="softmax"))
    model_no_dropout.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return (model_dropout, model_no_dropout)


def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    model_earlystopping = keras.Sequential()
    model_earlystopping.add(layers.Dense(50, activation="relu", input_dim=n_inputs))
    model_earlystopping.add(layers.Dense(100, activation="relu"))
    model_earlystopping.add(layers.Dense(n_outputs, activation="sigmoid"))
    model_earlystopping.compile(optimizer='rmsprop', loss='binary_crossentropy')
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)

    model_noearlystopping = keras.Sequential()
    model_noearlystopping.add(layers.Dense(50, activation="relu", input_dim=n_inputs))
    model_noearlystopping.add(layers.Dense(100, activation="relu"))
    model_noearlystopping.add(layers.Dense(n_outputs, activation="sigmoid"))
    model_noearlystopping.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return (model_earlystopping, dict(callbacks=[es]), model_noearlystopping, dict())



