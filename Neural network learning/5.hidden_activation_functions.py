'''
ACTIVATION FUNCTION
For an activation function, you can use a variety of different functions. To start with, we're just going to look at the step function.

STEP FUNCTION:
The idea is that if your input to this function is greather than 0, then your output will be 1. Otherwise your output is going to be a 0.

Now lets use this step function as an ACTIVATION function.
We use the step function after we do the inputs*weights + bias.
-> So that value of inputs*weights + bias is what is being fed through the activation function.
The main takeaway here is that the output is literally a 0 or a 1.

Now, the output that is a 0 or 1 becomes the input to the next neurons.

SIGMOID ACTIVATION FUNCTION
Context: Now, while you can train a neural network with the step function, it quickly became clear that using something like a sigmoid activation function is a little more easy and reliable to train a neural network due to the granularity of the output.

In this context, granularity means how detailed or fine-grained the output values can be.
The sigmoid function produces smooth, continuous values between 0 and 1 (like 0.23, 0.67, 0.89), while the step function only produces discrete jumps (just 0 or 1, nothing in between).
This smoothness gives the neural network more nuanced information during training, making it easier to adjust weights gradually and learn effectively.

We use the sigmoid activation function after we do the inputs*weights + bias.

'''