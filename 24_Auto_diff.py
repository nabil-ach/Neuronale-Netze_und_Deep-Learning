import tensorflow as tf
import torch

# TensorFlow Gradient Tape Example
a = tf.Variable(2.0)
b = tf.Variable(-3.0)
c = tf.Variable(10.0)
f = tf.Variable(-2.0)

with tf.GradientTape(persistent=True) as tape:
    e = a * b
    d = e + c
    L = d * f


print('a grad = ', tape.gradient(L, a))
print('b grad = ', tape.gradient(L, b))
print('c grad = ', tape.gradient(L, c))
print('f grad = ', tape.gradient(L, f))

print('\nd grad = ', tape.gradient(L, d))
print('e grad = ', tape.gradient(L, e))
print('L grad = ', tape.gradient(L, L))

print('\n---------------------------------------------------\n')
# PyTorch Autograd Example
f = torch.tensor(-2.0, requires_grad=True)
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)

e = a * b
d = e + c
L = d * f
L.backward()

print('a grad = ', a.grad)
print('b grad = ', b.grad)
print('c grad = ', c.grad)
print('f grad = ', f.grad)