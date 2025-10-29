# Layer Normalization

## What is Layer Normalization?

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
