import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input)
from tensorflow.keras.optimizers import *

from keras.applications.vgg16 import VGG16

import math

class LayerWiseLR(Optimizer):
    """
    Wrapper used to implement multipliers for the gradient for specific layers in the model
    Arguments:
        optimizer: instance of the optimizer
        multiplier: dictionary containing pairs of {layer name : multiplier}
        learning rate: initial learning rate of the optimizer
    """
    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LWLR", **kwargs):
        # checks for the presence of the _HAS_AGGREGATE_GRAD attribute,
        # present since version 2.11 with the introduction of the new optimizer APIs,
        # to determine how to initialize the wrapper instance
        if hasattr(Optimizer, "_HAS_AGGREGATE_GRAD"):
            # wrapper initialization with new APIs, with learning rate stored in an internal slot
            super().__init__(name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        else:
            # wrapper initialization with the old API, with the learning rate as an argument
            super().__init__(learning_rate, name, **kwargs)

        # storage of the attributes in the wrapper instance
        self._learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32) 
        self._optimizer = optimizer
        self._multiplier = multiplier

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        # initialization of the list containing the new variable-gradient pairs
        updated_grads_and_vars = []
        # iterates over each pair
        for grad, var in grads_and_vars:
            # get the current layer name, separating it from its type based on keras version
            layer_name = (var.path if hasattr(var, 'path') else var.name).split('/')[0]
            # apply the multiplier to gradient based on current layer
            # if no multiplier is associated, applies 1 as default value
            scaled_grad = grad * self._multiplier.get(layer_name, 1.0)
            # add the updated pair to the list
            updated_grads_and_vars.append((scaled_grad, var))
        # synchronize the learning rate of the base optimizer with that of the wrapper,
        # in order to apply changes due to callbacks, and then applies gradients
        self._optimizer.learning_rate.assign(self._learning_rate)
        self._optimizer.apply_gradients(updated_grads_and_vars)
        
    def _create_slots(self, var_list):
        # call the creation of the internal slots of the base optimizer
        self._optimizer._create_slots(var_list)

        
if __name__ == '__main__':
    # load VGG16
    model = VGG16(weights='imagenet')
    
    # instantiate the optimizer
    lr = 0.001
    opt = Adam(lr)

    # in this example the dict is built in such a way that for each layer before the dense section
    # a multiplier is associated with a value reduced to the previous one by a factor of root of two
    # alternatively, you can specify in the dict the name of the layer and the corresponding multiplier
    multiplier = {}
    new_keras = hasattr(opt, 'update_step')
    trainable = [(layer.path if new_keras else layer.name).split('/')[0] for layer in model.trainable_variables]
    # for each successive variable, i've a reduction by a factor of sqrt(2)
    current_mul = 1
    lr_factor = math.sqrt(2)
    # iterate over each trainable layer, skipping one (kernel and bias pairs)
    for layer in trainable[::2]:
        # get layer class name
        layer_type = model.get_layer(layer).__class__.__name__
        # if the current layer is a type on which we want to apply a multiplier
        if layer_type in ['Conv2D']:
            multiplier |= {layer : current_mul}
            current_mul /= lr_factor
    
    # use the wrapper to instantiate the optimizer and multiplier values for the learning rate
    opt = LayerWiseLR(opt, multiplier, lr)
