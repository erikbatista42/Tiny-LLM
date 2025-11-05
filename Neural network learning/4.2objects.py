import numpy as np
np.random.seed(0)

# Capital X signals that this is our exact training data set (machine learning standard)
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
    ]
# ^ We'll assume this is the input data to the neural network

'''
Now we will define our 2 "hidden layers". It's called "hidden" because it sits between the input and output, and we don't directly interact with it—only the neural network does during training.

We call it "hidden" because:
- We can see the INPUT layer (we provide the data)
- We can see the OUTPUT layer (we see the predictions)
- But the hidden layers are in the MIDDLE - we can't directly see or control what happens there.
- The neural network learns and adjusts the hidden layer weights automatically during training

---------
INPUT LAYER          HIDDEN LAYER(S)         OUTPUT LAYER
    ↓                     ↓                        ↓
[We provide]    [Hidden in the middle]      [We see results]
   Data         Neural network adjusts       Predictions
                    automatically
    
    😊                    🔒                       😊
  Visible               Hidden                  Visible
'''


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''we need to know 2 things:
            1. What is the size of the input coming into this layer?
            2. How may neurons do we want to have
        '''
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
# print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)

'''
Visual Summary:

INPUT DATA (X)
3 samples × 4 features
    ↓
┌──────────────────┐
│   LAYER 1        │
│   4 → 5 neurons  │  ← Hidden layer
└──────────────────┘
    ↓
3 samples × 5 features
    ↓
┌──────────────────┐
│   LAYER 2        │
│   5 → 2 neurons  │  ← Output layer
└──────────────────┘
    ↓
FINAL OUTPUT
3 samples × 2 predictions
'''