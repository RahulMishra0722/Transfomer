# Understanding Multi-Layer Perceptrons (MLPs)

## To Understand How Transformers work We Need to understand MLP'S (MULTI LAYERED PERCEPTRON);

## What is an MLP?

An MLP is a **feed-forward artificial neural network**. But what does "feed-forward" actually mean?

When you input data (pixels, text, numbers, etc.) into an MLP, it follows a single path:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer N → Output Layer
```

**Feed-forward** means the data flows in one direction only. Each layer:

1. Takes input from the previous layer
2. Performs computations (matrix multiplication, adds bias, applies activation functions)
3. Passes the result to the next layer
4. Produces a final output

This is different from RNNs, where outputs can feed back as inputs, and layers can communicate with themselves or other layers.

---

## MLP Architecture

### Input Layer

- Contains neurons that hold your input values (like pixel intensities or feature values)
- No computations happen here—it simply represents and passes data forward

### Hidden Layers

These are called "hidden" because they sit between the input and output layers. We can observe inputs and outputs, but the hidden layers work behind the scenes during inference.

**Where the magic happens:**

- Matrix multiplications (weights × inputs)
- Adding bias terms
- Applying activation functions
- Loss calculation

**Interesting fact:** The deeper we go into the network, the harder it becomes to understand what patterns individual neurons are learning. Deep layers learn increasingly abstract representations.

---

## Why Activation Functions Matter

**What if we didn't use activation functions?**

Without activation functions, no matter how many layers you stack, you'd just be combining linear transformations. Mathematically, multiple linear transformations collapse into a single linear transformation—essentially, you'd end up with one straight line.

**Why is this a problem?**

Think about representing complex objects like cats and cars:

- Can you capture all the features of a cat on a single line? No.
- Can you separate cats from cars using just a straight line? Not reliably.
- Do complex patterns fit into linear relationships? Rarely.

**The solution:** Activation functions introduce **non-linearity**, allowing the network to learn complex, curved decision boundaries and represent intricate patterns that exist in real-world data.

Common activation functions: ReLU, Sigmoid, Tanh, Leaky ReLU

---

## What are activation functions and why use activation functions at all?

Without **Activation Functions** You can stack as many Layers you want it will still collapse into a **Single Linear Transformation**, Why? supposed we had 3 layers and they all did = **W \* Input + some_bias** it all colapses down to Ax + b which is a straight line, But what does that mean? tho what does straight line prevent us from doing actually?

### Imagine this if you had cat data and then some dog related image data can you correctly represet that with a straight line no right?

so if you have a very specific problem in a very specific domain that doesnt have alot of tree nodes you can do away with **Activation Functions** But in transformers that do crazy stuff we need **Activation Function** To Introduce non linearty so it can learn complex patterns.

---

## Why Not Just Stack Infinite Layers?

If going deeper helps neurons learn more abstract patterns, why not add hundreds of layers?

**The problem: Diminishing returns**

Modern architectures like Transformers _do_ use many layers (GPT-3 has 96 layers!), but there are real challenges:

### 1. Overfitting

The model memorizes the training data instead of learning generalizable patterns. It performs perfectly on training data but fails on new, unseen data.

### 2. Vanishing/Exploding Gradients

As gradients backpropagate through many layers:

- **Vanishing:** Gradients become microscopic, making early layers barely learn
- **Exploding:** Gradients become massive, causing unstable training

This makes very deep networks difficult or impossible to train without special techniques (like residual connections, batch normalization).

### 3. Computational Cost

- More layers = more parameters
- More parameters = more memory, longer training time, higher costs
- Eventually, the performance gains don't justify the resource investment

---

## Key Takeaways

✓ MLPs process data in one direction (feed-forward)

✓ Activation functions are essential for learning complex patterns

✓ Hidden layers learn increasingly abstract representations

✓ Depth helps, but too many layers create training challenges

✓ Modern architectures use techniques like residual connections to train very deep networks successfully
