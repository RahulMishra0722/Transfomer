# Transformer V2: The "Why Does This Work?" Curriculum

## ~~1. The Adam Optimizer (Adaptive Moment Estimation)~~ [LEARNED!]

**The Core Mystery:** Standard SGD updates weights blindly based on the current gradient. But imagine a loss landscape shaped like a long, narrow canyon. SGD will bounce violently back and forth across the canyon walls while making agonizingly slow progress down the actual path.
**Before:** You used vanilla SGD (`weight = weight - learning_rate * gradient`). It blindly took a step in the direction of the gradient, applying the exact same step size to all 8 million parameters.
**After:** We switched to Adam. We allocated two new matrices (`m` and `v`) for every single weight in the model to track its personal historical momentum and variance, and mathematically adjusted the step size for every weight individually.
**The "Why" to investigate:**

- **Momentum (`m`):** Why did researchers steal the concept of "momentum" from physics? (Hint: A ball rolling down a canyon builds speed in the consistent direction and ignores the side-to-side bumps).
- **Variance (`v`):** Why do we need to track the squared gradients? (Hint: Some weights need to change fast, others need to change slow. Adam gives _every single weight_ its own custom speed limit).
- **Is it mathematically pure?** Nope! Adam is heavily empirical. They bolted physics equations onto statistics and found it worked better than anything else.

## 3. Global Gradient Clipping

**The Core Mystery:** If your network hits a weird patch of data, a gradient might "explode" to a massive number (like 50,000), corrupting your weights instantly. You used to clip each matrix individually to prevent this. Why is clipping individually mathematically wrong?
**Before:** You looped through each matrix independently. If a value in `W_Q` was > 1.0, you clipped it. If a value in `W_K` was fine, you left it.
**After:** We sum the squares of ALL gradients across the entire model first. If the _total_ global norm exceeds 5.0, we scale down every single matrix by the exact same proportion (e.g., multiply everything by 0.5).
**The "Why" to investigate:**

- Imagine you want to walk North-East (10 steps North, 10 steps East). If I clip your North movement to 2 steps, but leave East at 10, you are now walking almost entirely East! You changed the _direction_ of the vector.
- How does taking the $L_2$ norm (the Pythagorean theorem) of ALL 8 million gradients simultaneously allow us to shrink the step size without changing the angle we are walking?

## 4. Temperature Sampling

**The Core Mystery:** Softmax converts a bunch of raw numbers (logits) into percentages that add up to 100%. But sometimes the model is 20% sure it's 'A', 15% sure it's 'B', and 10% sure it's 'C'. It's too muddy.
**Before:** Your `sample_token` function just fed the raw logits directly into `softmax()`. If the model was unsure, it often picked hallucinated garbage text.
**After:** We pass `temperature = 0.8` into `sample_token` and divide every logit by 0.8 before calculating the softmax.
**The "Why" to investigate:**

- If you divide all the raw numbers by `0.8` _before_ doing the Softmax math, the percentages magically change to something like 60% for 'A', 5% for 'B', 1% for 'C'.
- Why does simple division artificially create extreme confidence? Look at the `exp()` function in the Softmax formula to see why spreading numbers apart creates exponential winners.

## 5. Generation Sliding Context Windows

**The Core Mystery:** Your model was trained to know what positions 0 through 127 mean. When generating text, if you feed it a 200-character string, it tries to do math on position 180. The math breaks.
**Before:** In `generate()`, you fed the entire `sequence` from `0` to `current_length` back into the model, meaning it was constantly trying to process text longer than `CONTEXT_LEN`.
**After:** We added a sliding window. The model now strictly slices the array to only look at the most recent `context_len` characters, forgetting the oldest ones.
**The "Why" to investigate:**

- How do production models like ChatGPT generate 4,000 words if they only have a context window of 8,000 tokens?
- They slide the window. They literally truncate the oldest text and pretend it doesn't exist. How does the model maintain long-term plot cohesion if it physically cannot see what it wrote 5 paragraphs ago? (This is one of the biggest active research problems in LLMs).

## ~~6. Learning Rate Warm-up and Decay~~ [LEARNED!]

**The Core Mystery:** If you keep your learning rate at `0.001` forever, your loss will drop quickly, but then it will get stuck and never reach the absolute bottom.
**Before:** Your LR schedule decayed immediately from Epoch 0 using `1.0 / (1.0 + 0.05 * epoch)`.
**After:** We added a "warm-up" phase. The LR stays perfectly flat for the first 5 epochs to let the model safely find its bearings in the loss landscape, and _then_ it begins a gentler decay.
**The "Why" to investigate:**

- Imagine dropping a bouncy ball into a bowl. If it keeps its kinetic energy (high learning rate), it will just bounce around the bottom of the bowl forever. You have to bleed off the energy (decay the learning rate) so it finally settles perfectly in the dead center.
- Why the "warm-up" phase? Why not start decaying immediately? (Hint: The neural network starts completely random. If you move too fast on step 1, you might permanently destroy the delicate initialization of the weights before they find a good path).

## 7. Byte-Pair Encoding (BPE) vs Character-Level Tokenization

**The Core Mystery:** Right now, your C model breaks text down into individual characters (A, B, C). This means the word "apple" takes up 5 full tokens in your 256-token context window. Modern LLMs like GPT-4 don't do this—they use sub-word chunks (like "app" and "le"). Why?
**The "Why" to investigate:**

- If you use character-level tokenization, your `vocab_size` is tiny (~65 characters), but your sequence gets massively long, wasting the Transformer's limited Attention span just figuring out how to spell words instead of understanding the sentence.
- If you use word-level tokenization, your sequence is short, but your `vocab_size` explodes to 100,000+ words, destroying your memory.
- How does Byte-Pair Encoding (BPE) find the perfect mathematical middle ground by automatically merging the most frequent character pairs (like "th", "er", "in") into their own custom tokens?
