# Multi-Head Attention

## The Core Problem: One Head = One Pattern

Imagine you have **only ONE attention head** for:

> "The fluffy cat sat on the mat"

### What Can One Head Learn?

Let's say it learns: **"Nouns should attend to verbs"**

- So "cat" looks at "sat" ✅

**But what about other useful patterns?**

- "cat" might also need to know WHERE it sat → should look at "mat"
- "cat" might need to know WHAT kind of cat → should look at "fluffy"
- "sat" might need to know WHO is sitting → should look back at "cat"

> **Key Insight:** One head can only learn one pattern well!

---

## The Solution: Multiple Heads = Multiple Patterns

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

---

## Mathematical Reason

```c
int head_dim = Q->cols / num_of_heads;
// If embedding = 512 and heads = 8
// Each head gets: 512 / 8 = 64 dimensions
```

### One Big Head (512 dims):

```
Q (512) · K (512) = one compatibility score
→ Tries to capture EVERYTHING in one number
→ Has to compromise between different patterns
```

### Eight Small Heads (64 dims each):

```
Head 1: Q (64) · K (64) = score for pattern 1
Head 2: Q (64) · K (64) = score for pattern 2
...
Head 8: Q (64) · K (64) = score for pattern 8

→ Each head specializes in ONE pattern
→ Concatenate all 8 outputs = 512 dims again
```

---

## The Rocket vs Cat Example

Suppose we want a head to learn about both rockets and cats. Do you think there are any commonalities between the two?

**NO!** And thus a head can't learn two things that have nothing in common.

**Why?** Can you put "cat" and "rocket" as a single vector and know which is which? No! It won't learn either - it will go all over the place.

Remember: **Learning means pointing in the same direction as something we're trying to learn in high-dimensional vector space.**

### One Head Trying Both:

```
Cat semantics: [furry, meows, pets, small, ...]
Rocket semantics: [space, propulsion, fast, metal, ...]

→ Creates a confused mess in 64 dimensions
→ Points nowhere useful
```

### Two Separate Heads:

```
Head 1: Specializes in animal relationships
  → Perfect at learning cat patterns

Head 2: Specializes in mechanical/vehicle relationships
  → Perfect at learning rocket patterns
```

---

## The Activation Function Analogy

This is like why we use activation functions to introduce non-linearity:

- A single line cannot learn complex patterns
- A single head can only learn one kind of information well
- We need multiple heads for complex patterns in GPT-like models

---

## The Final Output Projection (W_O)

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

## Summary: Multiple Heads = Multiple Patterns

- One head can only learn one pattern well
- Multiple heads can specialize in orthogonal concepts
- Like having multiple search queries running in parallel
- Each head gets a slice of the embedding dimensions
- Final output projection combines all perspectives

---

## TODO

- My understanding of multi head attention is somewhat flawed gotta fix it!
  But i wanna move on to the next thing for now
