'''
DERIVATIVES AND TANGENT LINES - Learning Step by Step

Goal: Understand what a derivative is and why it matters for neural networks.

SIMPLE EXPLANATION:
- DERIVATIVE = The slope (steepness) of a curve at one specific point
- TANGENT LINE = A straight line that just touches the curve at that point
- The derivative tells us: "If I change x a little, how much does y change?"
'''

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# STEP 1: Define a simple function (imagine this is your loss function)
# ============================================================================
def f(x):
    return 2*x**2

print("STEP 1: Our function is f(x) = 2*x²")
print("Let's see what y values we get for different x:")
print(f"  f(0) = {f(0)}")
print(f"  f(1) = {f(1)}")
print(f"  f(2) = {f(2)}")
print(f"  f(3) = {f(3)}")
print()

# ============================================================================
# STEP 2: Calculate the DERIVATIVE (slope) at a specific point
# ============================================================================
print("STEP 2: What's the slope at x=2?")
print("To find slope, we need two points very close together:")
print()

x_point = 2
tiny_step = 0.001  # Move just a tiny bit to the right

y1 = f(x_point)           # y at x=2
y2 = f(x_point + tiny_step)  # y at x=2.001

print(f"  Point 1: x={x_point}, y={y1}")
print(f"  Point 2: x={x_point + tiny_step}, y={y2}")
print()

# Calculate slope using: slope = rise/run = (y2-y1)/(x2-x1)
slope = (y2 - y1) / tiny_step
print(f"  Slope (derivative) = (y2-y1) / (x2-x1) = {slope:.2f}")
print()
print(f"  MEANING: At x=2, if we increase x by 1, y increases by ~{slope:.0f}")
print(f"           The curve is going UP steeply!")
print()

# ============================================================================
# STEP 3: What is a TANGENT LINE?
# ============================================================================
print("STEP 3: Draw a tangent line at this point")
print(f"A tangent line is a straight line that touches the curve at x={x_point}")
print(f"It has the same slope as the curve at that point: {slope:.2f}")
print()

# To draw a line, we need the equation: y = mx + b
# We know: m = slope, and the line passes through (x_point, y1)
# Solve for b: b = y - mx
b = y1 - slope * x_point
print(f"  Tangent line equation: y = {slope:.2f}*x + {b:.2f}")
print()

# ============================================================================
# STEP 4: VISUALIZE IT!
# ============================================================================
print("STEP 4: Let's see this on a graph!")

# Create the curve
x = np.arange(0, 5, 0.01)
y = f(x)

# Create the tangent line (just around our point)
x_tangent = np.arange(x_point - 1, x_point + 1, 0.1)
y_tangent = slope * x_tangent + b

# Plot everything
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Curve: f(x) = 2x²')
plt.plot(x_tangent, y_tangent, 'r--', linewidth=2, label=f'Tangent Line (slope={slope:.1f})')
plt.plot(x_point, y1, 'ro', markersize=15, label=f'Point ({x_point}, {y1})')

plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Derivative = Slope of Tangent Line', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.show()

# ============================================================================
# WHY THIS MATTERS FOR NEURAL NETWORKS
# ============================================================================
print()
print("=" * 60)
print("WHY THIS MATTERS:")
print("=" * 60)
print("In neural networks, we want to minimize LOSS.")
print("The derivative tells us which direction to adjust weights!")
print()
print("Example:")
print("  If derivative is POSITIVE → loss is increasing → move weight DOWN")
print("  If derivative is NEGATIVE → loss is decreasing → move weight UP")
print("  If derivative is ZERO → we're at the bottom! (optimal)")
print("=" * 60)



'''
So in a nutshell, we calculate how the loss changes when we adjust each weight (the derivative, which is the slope). Based on whether the derivative/slope is negative or positive, we adjust the weight in the right direction until we reach the minimum loss (derivative = 0), where we have optimal weights."
'''



'''
WHEN is this step done?
At the backward pass.


# STEP 1: FORWARD PASS
input_data = [1.0, 2.0, 3.0]
predictions = neural_network.forward(input_data)  # Get output
loss = calculate_loss(predictions, correct_answer)  # Loss = 0.35

# STEP 2: BACKWARD PASS - Calculate derivatives
derivatives = neural_network.backward(loss)  
# This calculates: ∂loss/∂weight1, ∂loss/∂weight2, ∂loss/∂weight3...

# STEP 3: UPDATE WEIGHTS
for weight, derivative in zip(weights, derivatives):
    weight -= learning_rate * derivative  # Adjust each weight

# STEP 4: REPEAT for next batch

-----------

After every forward pass, we:
Calculate the loss
Immediately calculate all derivatives (backpropagation)
Update all weights
This happens once per batch during training. If you have 1000 training samples and a batch size of 32, you'd calculate derivatives about 31 times per epoch (1000 ÷ 32).

'''