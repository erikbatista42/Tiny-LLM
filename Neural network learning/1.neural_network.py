# REPRESENTING A hidden layer of a neural network.


inputs = [1, 2, 3, 2.5] # each input is a neuron

# each neuron has its own weights

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]



# print(output)
'''
In a neuron, **inputs**, **weights**, and **bias** work together to produce an output. Here’s a breakdown of each part and how they're combined:

### 1. Input
* **Purpose:** This is the raw data or signal that you feed into the neuron.
* **Example:** If you have four inputs, your input data might be a list of numbers like `[1, 2, 3, 2.5]`, as shown in the video.

### 2. Weight
* **Purpose:** A weight is assigned to each input and determines its **importance** or **strength**.
    * A large positive or negative weight means that input will have a significant effect on the neuron's output.
    * A weight close to zero means that input will have very little effect.
* As the video explains, the weight acts as a **multiplier** for its corresponding input [05:30](http://www.youtube.com/watch?v=tMrbN67U9d4&t=330). During the "learning" process, the neural network "tunes" these weights to improve its accuracy [04:17](http://www.youtube.com/watch?v=tMrbN67U9d4&t=257).

### 3. Bias
* **Purpose:** The bias is a single, constant value that is added to the final sum. Its job is to **offset** the result [06:46](http://www.youtube.com/watch?v=tMrbN67U9d4&t=406).
* The video compares it to the 'b' in the equation of a line, `y = mx + b` [06:46](http://www.youtube.com/watch?v=tMrbN67U9d4&t=406), [21:00](http://www.youtube.com/watch?v=tMrbN67U9d4&t=1260). It allows the neuron to be "biased" towards firing or not firing. For example, a bias can help the neuron activate even if all its inputs are zero, or it can raise the threshold that the weighted inputs must overcome.

***

### How They Are Used Together

The core calculation of a neuron is to find the **weighted sum of its inputs and then add the bias**.

As the video you're watching explains, this process is: `inputs * weights + bias` [11:31](http://www.youtube.com/watch?v=tMrbN67U9d4&t=691).

This is done in three steps:
1.  **Multiply:** Each **input** is multiplied by its corresponding **weight**.
2.  **Sum (Dot Product):** All these multiplied values are added together. The video title, "[Neural Networks from Scratch - P.3 The Dot Product](http://www.youtube.com/watch?v=tMrbN67U9d4)", refers to this exact operation [12:12](http://www.youtube.com/watch?v=tMrbN67U9d4&t=732), [19:45](http://www.youtube.com/watch?v=tMrbN67U9d4&t=1185).
3.  **Add Bias:** The **bias** is added to this total sum [15:19](http://www.youtube.com/watch?v=tMrbN67U9d4&t=919).

The formula for a single neuron's output is:
`Output = (input₁ * weight₁) + (input₂ * weight₂) + ... + (inputₙ * weightₙ) + bias`

This final `Output` value is then typically passed to an "activation function" (which the video mentions around [06:27](http://www.youtube.com/watch?v=tMrbN67U9d4&t=387)) to produce the neuron's final signal.

---------

The Neuron = A small machine with:
1. External inputs (data flowing in from previous neurons)
2. Internal settings/knobs (weights and biases)
3. A computation (multiply inputs by weights, add bias)
An output (result sent to next neurons)


Other Neurons
     ↓↓↓ (inputs come in)
    ┌─────────────────┐
    │    NEURON       │
    │  ┌───────────┐  │
    │  │ weights:  │  │ ← These stay inside!
    │  │ biases:   │  │   They define THIS neuron's
    │  │ bias      │  │   behavior
    │  └───────────┘  │
    │  computation:   │
    │  output = inputs │
    │   × weights     │
    │   + bias        │
    └─────────────────┘
          ↓ (output goes out)
    To Next Neurons

'''

