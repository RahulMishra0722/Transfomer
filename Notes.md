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

# Understanding Attention Mechanisms: My Journey

## Table of Contents

1. [The Initial Confusion](#the-initial-confusion)
2. [The Big Realization](#the-big-realization)
3. [Direction vs Magnitude](#direction-vs-magnitude)
4. [How Learning Actually Works](#how-learning-actually-works)
5. [Understanding Q, K, V Matrices](#understanding-q-k-v-matrices)
6. [Why Scaling Matters](#why-scaling-matters)
7. [Multi-Head Attention](#multi-head-attention)

---

## The Initial Confusion

I always thought: **What the fuck are these Q, K, V matrices for?**

As someone who doesn't take anything as default, I cannot accept something until I intuitively see why it is the way it is. It was always strange to me:

- Why Q, K, V and not just 1 matrix?
- Why not Q, K, V, Z?

### The Realization

Then it hit me: **They just tried a bunch of stuff and Q, K, V happened to work best.**

I do experiments the same way - throw stuff together, keep what works, toss what doesn't. That's how research actually works!

> **Key Insight:** There's no secret mathematical proof. No divine insight. Just empirical results.

If A1, B2 worked better, we'd be using that instead. If just Q alone worked better, we'd use that. The fact that we use Q, K, V means people experimented and this configuration got the best results. That's it. No magic!

---

## Direction vs Magnitude

### The Misleading Example

When I first learned about attention, I was given this example:

**Example 1:** Similar vectors

```
A = [1, 2, 3]
B = [1, 2, 3]
Dot product = 14 (HIGH!)
```

**Example 2:** Different vectors

```
A = [1, 2, 3]
C = [0, 0, 0]
Dot product = 0 (LOW!)
```

**Conclusion shown:** "Similar vectors → HIGH dot product, Different vectors → LOW dot product"

### Why This Example Is Terrible

The `[0, 0, 0]` example created **three false impressions:**

1. ❌ Bigger numbers = more attention (because 14 > 0)
2. ❌ Only extreme cases matter (perfect match vs nothing)
3. ❌ Magnitude is what we're measuring (louder = more attention)

**All three are WRONG.**

### My Confusion

I kept thinking: "What if C = [30, 12, 15]?"

- That's way bigger than A = [1, 2, 3]
- Even after normalization, the ratios stay different
- So does bigger mean more attention or not?

The example made me think **magnitude = similarity**, which is backwards.

### What Actually Matters: Direction

```
A = [1, 2, 3]
B = [2, 4, 6]      // Same direction as A, just scaled up!
C = [3, 1, 2]      // Different direction

After normalization:
A_norm = [0.27, 0.53, 0.80]
B_norm = [0.27, 0.53, 0.80]  ← Identical to A!
C_norm = [0.65, 0.22, 0.43]  ← Actually different
```

See how C has different proportions? Vectors in the same direction (A and B) become identical after normalization.

### Attention Scores

- **A → B: HIGH** (same direction = pay more attention)
- **A → C: LOWER** (different direction = pay less attention)

> **The Real Insight:** Attention measures DIRECTION similarity (semantic meaning), not MAGNITUDE (loudness).

### The Complete Picture

**It's both, but direction matters more!**

We don't normalize in practice because:

- Magnitude can carry information (e.g., "this word is very relevant" vs "somewhat relevant")
- Scaling by `√d_k` is cheaper than normalizing each vector
- Empirically it works better this way

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

## Understanding Q, K, V Matrices

### The Context Problem

**Say we have a sentence:** "The cat sat on the mat"

Each word is represented as a vector, and these vectors are **context-independent**:

- The vector for "cat" doesn't know who is sitting
- The vector for "sat" doesn't know who is sitting or what they're sitting on
- The vector for "mat" doesn't know what's on it

**Attention solves this:** Each word looks at every other word and builds context-aware representations.

### Starting Point: Embeddings

```javascript
Word: "cat"
Embedding: [0.23, 0.91, 0.44, 0.12, ...] (512 numbers)
```

This is converting data into a high-dimensional array/vector (usually 512 dimensions).

---

### Query (Q): "What Should I Search For?"

```c
matmul(input, W_Q, Q)
```

**Example:**

```
"cat" embedding × W_Q = cat's Query vector
```

**W_Q learns to encode "Searching Behavior":**

- Maybe "cat" (noun) learns to search for verbs
- Maybe it learns to look for adjectives that describe it
- Model learns this by iterating and slowly descending towards the minima

**In other words:** Q defines what each word is looking for in other words.

---

### Key (K): "What Information Do I Advertise?"

```c
matmul(input, W_K, K)
```

**Example:**

```
"sat" embedding × W_K = sat's Key vector
```

**W_K learns to encode "I encode this type of info":**

- "sat" might advertise: "I'm a verb, I'm an action, I happened in the past"
- "mat" might advertise: "I'm a noun, I'm an object, I'm a location"

**In other words:** K defines what information each word makes available for searching.

---

### Value (V): "What Information Should I Pass Forward?"

```c
matmul(input, W_V, V)
```

**Example:**

```
"sat" embedding × W_V = sat's Value vector
```

**W_V learns to encode semantic meaning:**

- What information should be passed to the next layers if this word is attended to
- It's the refined, task-relevant information extracted from the word

**In other words:** V contains the actual useful information to retrieve when a word is deemed important.

---

### The Complete Flow

```
1. Compute Q, K, V for all words
   ↓
2. Calculate scores: Q × K^T
   (How much should cat attend to each word?)
   ↓
3. Scale by √d_k
   (Prevent gradient explosion)
   ↓
4. Apply softmax
   (Convert to probability distribution)
   ↓
5. Weighted sum: scores × V
   (Mix Value vectors based on attention)
   ↓
6. Output: context-aware representations
```

### Example: "cat" Attending to Other Words

```
Values from all words:
cat_Value = [0.5, 1.0, 1.5]
sat_Value = [2.0, 3.0, 4.0]  ← This gets "mixed in"
mat_Value = [1.0, 1.5, 2.0]

Attention scores for cat:
cat→cat: 0.1  (low self-attention)
cat→sat: 0.7  (HIGH - cat pays most attention to sat)
cat→mat: 0.2  (medium attention)

Result:
cat_new = 0.1 × cat_Value + 0.7 × sat_Value + 0.2 × mat_Value
        = [0.05, 0.1, 0.15] + [1.4, 2.1, 2.8] + [0.2, 0.3, 0.4]
        = [1.65, 2.5, 3.35]
```

The new "cat" vector now contains information about what action it's doing (sitting)!

### Signal vs Noise

Whatever is not relevant automatically fades:

- `cat→cat: 0.1 × vector` = small values (kills the noise)
- `cat→mat: 0.2 × vector` = medium values
- `cat→sat: 0.7 × vector` = large values (exaggerates the signal)

We drown out the noise and exaggerate the signal!

---

## Why Scaling Matters

### The Problem Without Scaling

```
Q·K with large dimensions → huge values

softmax([50, 1000, 10]) → [0, 1, 0]  ← One-hot pattern!
```

This loses all the meaning/scores because softmax saturates to one-hot.

### The Solution: Scale by √d_k

```
scaled_scores = (Q·K) / √d_k
              = (Q·K) / √512
              ≈ divide by ~22.6

softmax([2.2, 4.4, 0.4]) → [0.09, 0.82, 0.09]  ← Smooth distribution!
```

### Why This Works

- Without scaling → extreme softmax → gradients vanish → can't learn
- With scaling → smooth distribution → gradients flow → learning works

**It's about gradient preservation!**

---

## Multi-Head Attention

### The Core Problem: One Head = One Pattern

Imagine you have **only ONE attention head** for:

> "The fluffy cat sat on the mat"

#### What Can One Head Learn?

Let's say it learns: **"Nouns should attend to verbs"**

- So "cat" looks at "sat" ✅

**But what about other useful patterns?**

- "cat" might also need to know WHERE it sat → should look at "mat"
- "cat" might need to know WHAT kind of cat → should look at "fluffy"
- "sat" might need to know WHO is sitting → should look back at "cat"

> **Key Insight:** One head can only learn one pattern well!

### The Solution: Multiple Heads = Multiple Patterns

With **8 heads**, each can specialize:

```
Sentence: "The fluffy cat sat on the blue mat"

Head 1: "What does what?"
  cat → sat (0.9)  // What action is cat doing?

Head 2: "What describes this thing?"
  cat → fluffy (0.8)  // What kind of cat?
  mat → blue (0.7)    // What kind of mat?

Head 3: "What causes this action?"
  sat → cat (0.9)  // Who is sitting?

Head 4: "Where/what is the location?"
  sat → mat (0.8)  // Where is the sitting happening?
  on → mat (0.8)   // What is "on" referring to?

Head 5: "How are things spatially related?"
  cat → mat (0.7)  // Physical relationship chain
  sat → on → mat (chain of location)

Heads 6-8: Learn other patterns...
```

Each head discovers a **different linguistic or semantic pattern!**

### Mathematical Reason

```c
int head_dim = Q->cols / num_of_heads;
// If embedding = 512 and heads = 8
// Each head gets: 512 / 8 = 64 dimensions
```

#### One Big Head (512 dims):

```
Q (512) · K (512) = one compatibility score
→ Tries to capture EVERYTHING in one number
→ Has to compromise between different patterns
```

#### Eight Small Heads (64 dims each):

```
Head 1: Q (64) · K (64) = score for pattern 1
Head 2: Q (64) · K (64) = score for pattern 2
...
Head 8: Q (64) · K (64) = score for pattern 8

→ Each head specializes in ONE pattern
→ Concatenate all 8 outputs = 512 dims again
```

### The Rocket vs Cat Example

Suppose we want a head to learn about both rockets and cats. Do you think there are any commonalities between the two?

**NO!** And thus a head can't learn two things that have nothing in common.

**Why?** Can you put "cat" and "rocket" as a single vector and know which is which? No! It won't learn either - it will go all over the place.

Remember: **Learning means pointing in the same direction as something we're trying to learn in high-dimensional vector space.**

#### One Head Trying Both:

```
Cat semantics: [furry, meows, pets, small, ...]
Rocket semantics: [space, propulsion, fast, metal, ...]

→ Creates a confused mess in 64 dimensions
→ Points nowhere useful
```

#### Two Separate Heads:

```
Head 1: Specializes in animal relationships
  → Perfect at learning cat patterns

Head 2: Specializes in mechanical/vehicle relationships
  → Perfect at learning rocket patterns
```

### The Activation Function Analogy

This is like why we use activation functions to introduce non-linearity:

- A single line cannot learn complex patterns
- A single head can only learn one kind of information well
- We need multiple heads for complex patterns in GPT-like models

### The Final Output Projection (W_O)

```c
Matrix *final_result = matrix_alloc(input->rows, W_0->cols);
matmul(concat_result, W_0, final_result);
```

After concatenating 8 heads:

```
[head1: cat-action | head2: cat-description | head3: action-cause | ...]
```

These are 8 different perspectives, but they might have redundancy or conflicts.

**W_O learns to:**

1. Combine complementary info from different heads
2. Resolve conflicts (if head1 says X, head2 says Y)
3. Extract the most useful mixed representation

It's like having 8 witnesses describe a scene, then W_O writes the coherent summary!

---

## Summary: What I Learned

### 1. Q, K, V Are Empirical

- They exist because they worked in experiments
- Not because of mathematical necessity
- If something else worked better, we'd use that

### 2. Attention Measures Semantic Similarity

- Are these words related in meaning?
- **Direction matters more than magnitude** (but both matter!)
- We use raw Q·K (not normalized) to preserve magnitude info

### 3. Scaling Prevents Gradient Death

- Large dot products → saturated softmax → one-hot → can't learn
- Scaling by √d_k keeps distributions smooth → gradients flow

### 4. Learning Is Alignment Through Optimization

- Gradient descent slowly adjusts weights
- Relevant Query-Key pairs naturally get high dot products
- It's emergent behavior, not a conscious "choice"

### 5. Multiple Heads = Multiple Patterns

- One head can only learn one pattern well
- Multiple heads can specialize in orthogonal concepts
- Like having multiple search queries running in parallel

### 6. Signal Amplification + Noise Suppression

- Softmax creates probability distribution (sums to 1)
- High attention scores get exaggerated
- Low attention scores get suppressed
- This preserves dynamic relationships between words

---

## The Magic Disappeared

And that felt good - because now I actually know **WHY** it works, not just that it does.

The magic disappeared when I understood the math.

--TODO --

- My understanding of multi head attention is somewhat flawed gotta fix it!
  But i wanna move on to the next thing for no
