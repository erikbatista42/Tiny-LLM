'''
**ACTIVATION FUNCTIONS**
For an activation function, you can use a variety of different functions. 

STEP FUNCTION:
To start with, we're just going to look at the step function.
The idea is that if your input to this function is greather than 0, then your output will be 1. Otherwise your output is going to be a 0.

Now lets use this step function as an ACTIVATION function.
We use the step function after we do the inputs*weights + bias.
-> So that value of inputs*weights + bias is what is being fed through the activation function.
The main takeaway here is that the output is literally a 0 or a 1.

Now, the output that is a 0 or 1 becomes the input to the next neurons.

----------

SIGMOID FUNCTION
Context: Now, while you can train a neural network with the step function, it quickly became clear that using something like a sigmoid activation function is a little more easy and reliable to train a neural network due to the granularity of the output.

In this context, granularity means how detailed or fine-grained the output values can be.
The sigmoid function produces smooth, continuous values between 0 and 1 (like 0.23, 0.67, 0.89), while the step function only produces discrete jumps (just 0 or 1, nothing in between).
This smoothness gives the neural network more nuanced information during training, making it easier to adjust weights gradually and learn effectively.

We use the sigmoid activation function after we do the inputs*weights + bias.

----------

RECTIFIED LINEAR UNIT FUNCTION
This function is very simple. If input or x > 0, the output = x. If x <= 0, the output is 0.
Like the sigmoid activation function, the output could be a little granular. It can't be less than 0 so if enough things are 0 it will output to 0.

We use the rectified linear activation function after we do the inputs*weights + bias.

--------------------------------------------------
Why would we use the rectified linear function over the sigmoid function which also produces a granular ouput?
-> The sigmoid problem has an issue reffered to as the vanishing gradient problem. 

3 main reasons why we would be using rectified linear:
    1. It's still granular so we can still optimize well with it.
    2. It's fast because it's a very simple calculation.
    3. It simply works.

Have been learning today more about how the ReLU works with neural networks. A better and visual explanation here: https://claude.ai/public/artifacts/b10a161b-6e5e-49f9-9ba3-db5d09f8f04d
'''


'''
HOW SQUIGGLY LINES ARE MADE ON THE GRAPH
The very first node is the input and the last one is the output, creating (x,y) coordinates for the graph.
Each time the neural network runs, it creates a dot in the graph. After some runs, you may see a squiggly line if you draw it out rather than a straight line.

More detailed:
A neural network takes an input value (X) and runs it through layers of neurons. Each neuron calculates Y = (input × weight) + bias, then applies ReLU which outputs max(0, Y). In Layer 1, all 14 neurons look at the same input but produce different outputs because each has different weights and biases, causing them to "turn on" at different input values. Layer 2 takes all those Layer 1 outputs, multiplies each by its own weight, adds them together with a bias, and applies ReLU again. This process repeats until you reach the final output neuron. When you plot the input (X) against the final output (Y) for every possible input value from -3 to +3, you get a squiggly curve because neurons constantly turn on and off at different points, changing their contribution to the final output. The weights and biases control this behavior, and adjusting them changes the shape of the curve—which is exactly what "training" a neural network means: tweaking these values until the network produces the outputs you want for given inputs.

Learn more here: https://claude.ai/share/4498c203-35b2-4f2c-ac9b-dc581ac94680
'''

import numpy as np

np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# linear activation function
for i in inputs:
    output.append(max(0, i))
print(output)