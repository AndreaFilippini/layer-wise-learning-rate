import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input)
from tensorflow.keras.optimizers import *

from keras.applications.vgg16 import VGG16

import math

class Multiplier(Optimizer):
    """
    Wrapper used to implement multipliers for the gradient for specific layers in the model
    Arguments:
        optimizer: instance of the optimizer
        multiplier: dictionary containing pairs of {layer name : multiplier}
    """
    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LRM", **kwargs):
        if hasattr(optimizer, 'update_step'):
            super().__init__(learning_rate, **kwargs)
        else:
            super().__init__(name, **kwargs)
            self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))            
        self._optimizer = optimizer
        self._multiplier = multiplier

    # function used to apply a multiplier to a specific layer
    def mul_param(self, param, var):
        # get layer name
        layer_key = var.name.split('/')[0]      
        # if there's a multiplier value associated with the layer, apply it to the parameter
        if layer_key in self._multiplier:
            param *= self._multiplier[layer_key]
        return param
            
    # update step used in keras 3.X
    def update_step(self, grad, var, learning_rate):
        new_lr = self.mul_param(self._optimizer.learning_rate, var)
        self._optimizer.update_step(grad, var, new_lr)

    def build(self, var_list):
        super().build(var_list)
        self._optimizer.build(var_list)
        
    # update step used in keras 2.X
    @tf.function
    def _resource_apply_dense(self, grad, var):
        self._optimizer._resource_apply_dense(self.mul_param(grad, var), var)

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list)
        
if __name__ == '__main__':
    # load VGG16
    model = VGG16(weights='imagenet')
    
    # load the optmizer
    opt = Adam(0.001)

    # in this example the dict is built in such a way that for each layer before the dense section
    # a multiplier is associated with a value reduced to the previous one by a factor of root of two
    # alternatively, you can specify in the dict the name of the layer and the corresponding multiplier
    multiplier = {}
    new_keras = hasattr(opt, 'update_step')
    trainable = [(layer.path if new_keras else layer.name).split('/')[0] for layer in model.trainable_variables]
    # for each successive variable, i'll have a reduction by a factor of sqrt(2)
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
    
    # use the wrapper to instantiate the optimizer and multiplier values
    opt = Multiplier(opt, multiplier)
