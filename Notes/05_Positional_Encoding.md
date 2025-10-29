# Positional Encoding

## Why use Positional Encoding at all?

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
