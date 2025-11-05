import numpy as np

# the dot product just means multiple pairs, then add them all up.

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]

bias = 2
output = np.dot(weights, inputs) + bias

print(output)