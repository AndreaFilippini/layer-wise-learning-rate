# Layer Wise Learning Rate
Layer wise learning rate for tensorflow/keras that applies different learning rate values to different layers of a neural network

# Examples
Once the optimizer is encapsulated inside the 'LayerWiseLR' wrapper class, you can assign to each layer a multiplier that will be applied to the learning rate during the optimization

**Assigns to dense layers of the vgg network two different multipliers**
```python
lr = 0.001
multiplier = {'fc1' : 0.9, 'fc2' : 0.4}
opt = LayerWiseLR(Adam(lr), multiplier, lr)
```
