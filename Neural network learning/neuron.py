# REPRESENTING A NEURON

# These are values coming in from other neurons. Each index is a representation of a neuron output.
inputs = [1.2 , 5.1, 2.1]
weights = [0.1, 0.2, 0]

# This value comes from the neuron itself. It does not come from other neurons.
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias

print(output)