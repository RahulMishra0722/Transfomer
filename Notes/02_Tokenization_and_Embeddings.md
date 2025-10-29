# Tokenization and Embeddings

## Now Since we will be Learning Transformers Lets get the basics out:

## Tokenization

**What is Tokenization?**
Converting text into numbers a computer can understand.
So to tokenize the data we are traning on or inferencing we have come up with a strategy to tokenze the data

**The Easiest way would be**
We take the inputs we assign an id to each uinique character

### 1. We first from the data find all unique chars

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

### 2. Now Create a look up table for each ASCII

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

---

## Embeddings

But These still numbers how do extract meaning / assign meaning Well how i dont understand and still dont get it lets understand there are these things called `"Embeddings Table"`.

### What is an Embedding Table?

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

### Starting Point: Embeddings

```javascript
Word: "cat"
Embedding: [0.23, 0.91, 0.44, 0.12, ...] (512 numbers)
```

This is converting data into a high-dimensional array/vector (usually 512 dimensions).
