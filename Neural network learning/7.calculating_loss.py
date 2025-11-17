'''
There are many different ways to calculate loss. We're going to use "Categorical Cross-Entropy" formula to calculate loss.

One-hot encoding
The idea is that you'll have a vector that is n-classes long. That vector is filled with 0s except at the index of the target class where you'll have a 1.

Classes: 5
Label: 2
One-hot[0, 0, 1, 0 ,0]
'''

'''
Logarithm

solving for x
e ** x = b
-> b=5.2
   print(np.log(b))
'''
import numpy as np
import math

softmax_output = [0.7, 0.1, 0.2] # example outputs you might get from the softmax activation function
target_output = [1, 0,0]

loss = -(math.log(softmax_output[0])*target_output[0] +
        math.log(softmax_output[1])*target_output[1] +
        math.log(softmax_output[2])*target_output[2]
)
# print(loss)

print(-math.log(0.7))
print(-math.log(0.5))



# 
# 
# 

'''
How Categorial-Cross Entropy Loss works

---------------------------------------------

STEP 1: Understanding the setup

You have:
Softmax output (model's prediction as probabilities)
-> [0.7, 0.1, 0.2]
• Class 0: 70% confident
• Class 1: 10% confident
• Class 2: 20% confident

Target output(correct answer in one-hot format)
-> [1, 0, 0]
• The correct class is 0

---------------------------------------------

STEP 2: What is One-Hot Encoding?
It's a way to represent the correct answer:
Classes: [Dog, Cat, Human]
Correct answer: Dog (index 0)
One hot: [1, 0, 0] <- one marks the correct class

---------------------------------------------

STEP 3: The Loss formula (Categorical Cross-Entropy)
Loss = -(log(pred[0])*target[0] + log(pred[1])*target[1] + log(pred[2])*target[2])

Let's plug in the values:
Loss = -(log(0.7)*1 + log(0.1)*0 + log(0.2)*0)
     = -(log(0.7)*1 + 0 + 0)
     = -log(0.7)
     = 0.357
Notice: Because of the one-hot encoding(only one "1", rest are "0s"), only the predicted probability for the correct class matters!

---------------------------------------------

STEP 4: WHY use -log()?
The negative log creates a penalty that:

High confidence + correct = Low loss(Good!)
-log(0.7) = 0.357 <- Model was 70% sure and correct -> Small loss
-log(0.9) = 0.105 <- Model was 90% sure and correct -> Even smaller loss

Low confidence + correct = High loss (Bad!)
-log(0.5) = 0.693 <- Model was only 50% sure (unsure) -> Bigger loss
-log(0.1) = 2.303 <- Model was only 10% sure (very wrong!) -> huge loss
-log(0.01) = 4.605 <- Model was only 1% sure -> massive loss

STEP 5: The magic of -log()
As the model's confidence in the correct answer decreases, the Why This Works
✅ Rewards confidence in correct answers (low loss)
✅ Punishes low confidence in correct answers (high loss)
✅ The penalty grows exponentially as confidence drops
✅ Simple to calculate and great for training neural networks
That's it! The loss tells the model "how wrong you are" so it can learn and improve.
'''