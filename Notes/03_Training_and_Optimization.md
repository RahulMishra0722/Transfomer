# Training and Optimization

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

## How Learning Actually Works

### The Initial Claim I Struggled With

> "A 'louder' key can get more attention even if misaligned, and a 'quieter' key might get less even if well-aligned. The model learns whether to use magnitude or direction by learning Q/K/V matrices that encode the right information."

### My Problem

I couldn't just accept: **"The model learns whether to use magnitude or direction"**

How could it possibly learn to pick one over the other? In math, it didn't sound like that should be possible.

### The Breakthrough: Backpropagation

Then I remembered what learning actually is:

1. **Gradient descent**
2. **Calculate loss** - see how wrong we were
3. **Update the weights** and pass it back so other layers can fix their mistakes
4. **Keep iterating** until we stop learning or improvements are tiny

### What It Does Internally

It slowly adjusts the weights. And **changing the matrix weights changes its direction in vector space**, which changes what parts are similar to what.

These slow changes eventually make them similar.

### The Key Insight

> **The model doesn't learn to "pick" one or the other.**
>
> **It just points in the same direction slowly and steadily, which eventually causes them to align!**

Through backprop, gradient descent slowly adjusts W_Q and W_K so that relevant Query-Key pairs naturally have high dot products. It's not a conscious "choice" - it's emergent from optimization!

---

## Summary

| Technique               | Use Case            | Key Benefit                                    |
| ----------------------- | ------------------- | ---------------------------------------------- |
| **Mini-batch SGD**      | Large datasets      | Incremental weight updates, efficient training |
| **Layer Normalization** | Transformers, RNNs  | Stable training, works with any batch size     |
| **Batch Normalization** | CNNs (images/video) | Fast convergence for grid-like data            |
