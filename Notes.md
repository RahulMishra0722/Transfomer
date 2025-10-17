# Transformers Architecture

## To Understand How Transformers work We Need to understand MLP'S (MULTI LAYERED PERCEPTRON);

# Understanding Multi-Layer Perceptrons (MLPs)

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

## Now Since we will be Learning Transformers Lets get the basics out:

### Tokenization

**What is Tokenization?**
Converting text into numbers a computer can understand.
So to tokenize the data we are traning on or inferencing we have come up with a strategy to tokenze the data
**The Easiest way would be**
We take the inputs we assign an id to each uinique character

1. We first from the data find all unique chars

```c
  int seen[256] = {0};
  int unique_count = 0;
  for (const char *p = text; *p; p++)
  {
    unsigned char c = (unsigned char)*p;
    if (seen[c] == 0)
    {
      seen[c] = 1;
      unique_count++;
    }
  }
```

2. Now Create a look up table for each ASCII

```c
  int idx = 0;
  for (int i = 0; i < 256; i++)
  {
    if (seen[i])
    {
      vocab->chars[idx] = (char)i;
      vocab->char_to_id[i] = idx;
      idx++;
    }
  }

```

On Each Correct ASCII Position `vocab->char_to_id[i] = idx;` assign the id of the chars so if now i pass 65 / `a` lower case `char_to_id[i]` this will give the index of where the charcater is stored used for tokenization and detokenization.

But These still numbers how do extract meaning / assign meaning Well how i dont understand and still dont get it lets understand there are these things called `"Embeddings Table"`.

#### What is an Embedding Table?

First of these idiots have to be removed from naming things its fucking confusing it basically a fancy word for taking any sort of data and `converting to a vector represenation` of the data
You are essensialy placing the output from tokenization and put it into a **higer dimentional space** on their ASCII positions.

```c
Matrix *embed_tokens(int *token_ids, int seq_len, Matrix *embedding_table)
{
  Matrix *matrix = matrix_alloc(seq_len, embedding_table->cols);
  for (int row = 0; row < seq_len; row++)
  {

    int token_id = token_ids[row];
    for (int col = 0; col < embedding_table->cols; col++)
    {
      matrix->data[row * embedding_table->cols + col] = embedding_table->data[token_id * embedding_table->cols + col];
    }
  }
  return matrix;
}
```

**So what doe this do?**
This basically gets a already filled embedding_table which has random values between -0.01 - 0.01 and for context the dimention of the embedding_table is vacab_size `*` d_model(dimentions of the model)
Whats the goal? we basically create a new matrix of the dimention described above and copy over the tokens from token ids which basically contain the id of the characters that we calculated during during tokenization and assign in those id position rows `embedding_table->data[token_id * embedding_table->cols + col]` `token_id === row` i still dont know why we do it this way

## Gradient Descent: Why Stochastic?

### What gradient descent version will we use?

We will use some version of **Stochastic Gradient Descent (SGD)** as normal gradient descent sucks.

### Why does normal gradient descent suck?

Let's take an example to hit this home:

**Example:** If you had 10 million data points, what normal gradient descent will do is:

- Go through the **entire dataset** (all 10 million points)
- Update weights **only at the end**
- This is **capital intensive** and **really time consuming**
- Cannot be used on big datasets

### So what do we use and why?

We use **Stochastic Gradient Descent (SGD family)**.

**Why?** What makes this better?

- It does one thing that makes this both **cost** and **time efficient**

### What does it do?

**The process:**

1. At the beginning of each training cycle (and even in between each loop), it **shuffles the data**
2. Picks up a portion of **fixed window size** (`i` to `n`) where:
   - Window size is **fixed** across loops
   - The **data** constantly changes within the window each loop
3. Unlike normal gradient descent, we:
   - Calculate gradient descent on these `n - i` values
   - **Instantly update the weights** each cycle
   - Weights learn **as they're being trained**, not at the end

**Example:**

- 10 million data points
- Window size: 1,000
- Result: **10,000 weight updates** (instead of 1)

---

## Layer Normalization

### What is Layer Normalization?

### The Core Problem It Solves

During training, the **distribution of inputs from previous layers changes after every loop**, leading to:

#### Problems Without Normalization:

1. **Data distribution becomes very random**

   - Bigger values keep getting bigger
   - Smaller values keep getting smaller
   - We **lose information** in the data
   - Training won't produce any outputs
   - Makes it harder for learning values and other helper fields to do their job

2. **Numerical instability problems**
   - **Explosion:** Float can handle `~±10^38`
     - Keep multiplying numbers → eventually exceed `~±10^38`
     - Results in **memory error** or **NaN errors**
   - **Vanishing:** Going too far in negative direction
   - Apart from things losing meaning, we also have these catastrophic failures

### Why We Should Use It (Pros):

1. **Faster training**

   - Data has a range it fits between
   - Variance decreases
   - Better learning overall

2. **Doesn't require sophisticated learning rates**

   - Can work with random big ones

3. **Better learning since data is normalized**
   - Data is no longer all over the place

#### Visual Intuition:

I see this as a graph in my head:

- **Without normalization:** Points at random places, sparse graph, data all over the place
  - When trying to train, won't cover all information properly
  - Have to look all over the place
- **With normalization:** Known and smaller surface area to learn on
  - Data doesn't get lost

---

## Why Layer Normalization and Not Others?

### Layer Norm vs Batch Norm

#### Layer Normalization:

- Goes **vertically/horizontally** (however you imagine a vector)
- Looks at what's in the **current layer/vector** and processes that
- **Key advantage:** Don't need to rely on other vectors/layers for normalizing data
- **Works with any data size**

#### Batch Normalization:

- **Released:** 2015
- **Designed for:** CNN models (Convolutional Neural Networks)
  - Works with grid-like data, specifically **images and videos**
  - Worked great for this use case

**What happened in 2016:**

- People tried Batch Norm in **RNN** (Recurrent Neural Network)
  - RNNs are designed to process **sequential data** (text, speech, time-series)
  - Maintains "memory" of previous inputs to influence current/future outputs
  - Examples: ChatGPT, Claude, Grok, Gemini, etc.
- **Result:** It sucked really bad
  - Didn't work well with small data
  - Needed to keep copies of statistics from training phase around
  - People didn't like how sucky and unreliable it was

**The Switch:**

- People changed approaches and switched to **Layer Norm**
- It works great!

---

## Summary

| Technique               | Use Case            | Key Benefit                                    |
| ----------------------- | ------------------- | ---------------------------------------------- |
| **Mini-batch SGD**      | Large datasets      | Incremental weight updates, efficient training |
| **Layer Normalization** | Transformers, RNNs  | Stable training, works with any batch size     |
| **Batch Normalization** | CNNs (images/video) | Fast convergence for grid-like data            |

---

# What are activation functions and why use activation functions at all?

Without **Activation Functions** You can stack as many Layers you want it will still collapse into a **Single Linear Transformation**, Why? supposed we had 3 layers and they all did = **W \* Input + some_bias** it all colapses down to Ax + b which is a straight line, But what does that mean? tho what does straight line prevent us from doing actually?

## Imagine this if you had cat data and then some dog related image data can you correctly represet that with a straight line no right?

so if you have a very specific problem in a very specific domain that doesnt have alot of tree nodes you can do away with **Activation Functions** But in transformers that do crazy stuff we need **Activation Function** To Introduce non linearty so it can learn complex patterns.

## 1. Positional Encoding

### Why use Positional Encoding at all?

Based on the paper "Attention is All You Need," it's clear that **attention is the most crucial part of a Transformer model**.

When I implemented the attention function following the paper, I just did:

- Dot products
- Transposing matrices
- Some fancy additions and softmax
- That was it

### The Problem with Just Attention

Let's think about this in the context where transformers need to **generate text**. To generate text, you first need to **understand what text means**.

**What I mean by "understanding text":**

English (or any language) is hard to represent as math because **one word put before or after can mean completely different things**.

**Example:**

- "I hit a dog with a snowball"
- "I saw a dog stepped on a snowball"

The main focus is still the **dog** and the **snowball**, but they mean **completely different things**!

### The Core Realization

We can clearly say that **just multiplying matrices together won't preserve the word's position** - like where the word exists in the sentence.

**Position makes all the difference!**

Think about it:

- "Dog hit snowball"
- "Snowball hit dog"

Same words, completely different meanings! The attention mechanism I built has **NO way to tell these apart** because it only looks at what the words are, not where they are.

### What's Missing?

My attention function is **permutation invariant** - fancy way of saying: if you shuffle the input words around, attention wouldn't even notice!

It's like:

- You have all the puzzle pieces (the words)
- But you lost the picture on the box (the order)
- You know what pieces you have, but not how they fit together

### This is Where Positional Encoding Comes In

Positional Encoding solves this by adding **position information** to each word embedding:

- Word at position 0 gets a unique mathematical signature
- Word at position 1 gets a different signature
- Word at position 2 gets another different signature
- And so on...

Now when attention does its matrix multiplications, it's working with:

- **Word meaning** (from embeddings) + **Position info** (from positional encoding)

The model can finally tell: "This word is at the START of the sentence" vs "This word is at the END of the sentence"

**Without this, the transformer is blind to word order - it's like trying to read a book with all the pages shuffled!**

### How Does It Work?

The paper uses sine and cosine functions to generate these position signatures:

- **For even dimensions (0, 2, 4, ...):** Use `sin(position / 10000^(dimension/d_model))`
- **For odd dimensions (1, 3, 5, ...):** Use `cos(position / 10000^(dimension/d_model))`

**Why sine and cosine?**

- They create smooth, periodic patterns
- Each position gets a unique "fingerprint"
- The model can learn to recognize positions and their relationships
- Example: It can learn "position 5 is 2 steps after position 3"

### What You Need to Do

Create a function that:

1. Takes your input embeddings (seq_len × d_model)
2. For each position and each dimension, calculate the sin/cos value
3. **Add** these values to the embeddings (element-wise addition)
4. Feed the result to the encoder

Now your transformer can finally understand that word order matters!

```

```
