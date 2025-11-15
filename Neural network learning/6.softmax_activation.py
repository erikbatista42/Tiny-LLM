'''
SOFTMAX ACTIVATION
The reason we're exploring another activation function is because there are some problems with the ReLU function. To name a few:
2. Unbounded - Because it grabbing the max(0, y), y can be anything. therefore this is unbounded.
3. We don't really have a way to determine how wrong this is in per sample sample that comes in.
With all these reasons, it is hard to calculate loss.
-> So for example, if we have [0, 5.2, 0, 12.8, 3.1] - these numbers don't give you a clear measure of "how arong is this"? 
    • With Softmax you get probabilities [0.01, 0.15, 0.02, 0.80, 0.02]  (all sum to 1.0)
    • We can see the model is 80% confident in feature #4.
'''


'''
Exponential function
y=e^x
e = ~2.7182...
x = the act of applying the exponential function to some value

This function solves our negative number issues by making sure no values can be negative
'''
import math
layer_ouputs = [4.8, 1.21, 2.385]

# E = 2.71828182846
E = math.e

exp_values = []

for output in layer_ouputs:
    exp_values.append(E**output)

print(exp_values)

# NORMALIZATION - a single value divided by the total of all the values
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values))