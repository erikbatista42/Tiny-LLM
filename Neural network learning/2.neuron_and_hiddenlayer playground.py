# REPRESENTING A NEURON

# These are values coming in from other neurons. Each index is a representation of a neuron output.
inputs = [1.2 , 5.1, 2.1]
weights = [0.1, 0.2, 0]

# This value comes from the neuron itself. It does not come from other neurons.
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias

print(output)



# ------------------------------------------------------------

# REPRESENTING A hidden layer of a neural network.


# inputs = [1, 2, 3, 2.5] # each input is a neuron

# # each neuron has its own weights
# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# # This value comes from the neuron itself. It does not come from other neurons.
# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
#           inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
#           inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3,
# ]

# print(output)