#!/usr/bin/env python3

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# The tensor contains its gradient function
print (f"layer result z = x * w+ b = {z}")
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
print(f"Loss: {loss}")

# Calculate the loss gradient wrt params
loss.backward()
print(f"w.grad = {w.grad}")
print(f"b.grad = {b.grad}")