# Layer Wise Learning Rate
Layer wise learning rate for tensorflow/keras that applies different multipliers to the gradients of different layers of a neural network.

# Examples
Once the optimizer is encapsulated inside the 'Multiplier' wrapper class,, you can assign to each layer a multiplier that will be applied to the gradient during the optimization

**Assigns to dense layers of the vgg network two different multipliers**
```python
multiplier = {'fc1' : 0.9, 'fc2' : 0.4}
opt = Multiplier(Adam(0.001), multiplier)
```
