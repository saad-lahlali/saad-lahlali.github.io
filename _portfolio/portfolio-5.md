---
title: "Beyond a Single Input: How I Built a Multi-Network Model to Predict Professions from Biographies"
excerpt: "Can a machine guess a person's profession just by reading their bio? This post details how I tackled this complex multi-label classification challenge. I designed a unique multi-input neural network in Keras that combines the general knowledge of pre-trained GloVe embeddings with the specific insights from trainable embeddings. Discover the architecture that achieved an impressive 0.79 F1 score and see how a simple probability threshold made all the difference."
collection: portfolio
---

-----

### Combining pre-trained and trainable embeddings in Keras to achieve a 0.79 F1 score in a complex multi-label classification task.

Can you guess someone’s profession just by reading their bio? You probably can. A bio that mentions "leading roles" and "box office hits" screams 'Actor', while one discussing "groundbreaking research" and "academic journals" points to a 'Researcher'.

For a human, this is intuitive. But can we teach a machine to do the same? This was the challenge I took on: building a model to predict a person's occupation—or multiple occupations—from their Wikipedia summary.

This isn't a simple classification problem. A person can be more than one thing. For example, Arnold Schwarzenegger is both an 'Actor' and a 'Politician'. This makes it a fascinating **multi-label text classification** task.

My approach involved creating a custom neural network with multiple inputs to capture different layers of meaning from the text. By the end, this hybrid model achieved an F1 score of **0.793**, proving that a more complex architecture can unlock significant performance gains. Here’s how I did it.

## The Data and the Challenge

The goal was to predict one or more of 20 possible occupations (from 'Politician' to 'Screenwriter') based on the summary text of a person's Wikipedia page.

The core challenge, as mentioned, is its multi-label nature. A simple `softmax` output, which is great for single-label problems, wouldn't work here. Instead, the model needs to output an independent probability for each of the 20 occupations.

## The Foundation: GloVe and Word Embeddings

Before a model can "understand" text, we need to convert words into numbers. This is done using **word embeddings**, which are dense vector representations of words that capture their semantic meaning.

Instead of training these from scratch, I decided to use **GloVe (Global Vectors for Word Representation)**. GloVe embeddings are pre-trained on a massive corpus of text (in my case, 6 billion tokens from Wikipedia and Gigaword). Using them is like giving your model a head start—it already has a general understanding of the English language before seeing a single sample from our dataset.

The first step was to load the 300-dimensional GloVe vectors and create an `embedding_matrix`. This matrix maps each word in my project's vocabulary to its corresponding GloVe vector.

```python
# Create a vocabulary from the training data
vocab_to_id = load_vocabulary(vocab_file)
num_tokens = len(vocab_to_id)

# Load pre-trained GloVe embeddings
path_to_glove_file = "./glove.6B.300d.txt"
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

# Create the embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

## The Core Idea: A Multi-Input Network Architecture

Here’s where things get interesting. Instead of feeding the text into a single network pipeline, I designed a model with **three parallel input streams**. The idea was to process the same text in three different ways and then merge the findings to make a final, more informed decision.

Why do this?

1.  **Combine General and Specific Knowledge:** One stream uses the pre-trained GloVe embeddings (general knowledge), while the other two learn their own embeddings from scratch, making them highly specific to this task.
2.  **Ensemble Effect:** Using multiple streams with slightly different structures acts like a mini-ensemble within a single model, often leading to more robust predictions.

Here is a simplified look at the architecture defined in the `create_model_2` function:

```
Input Text (e.g., a Wikipedia summary)
     |
     +------------------+------------------+
     |                  |                  |
[Network 1]        [Network 2]        [Network 3]
- Non-Trainable    - Trainable        - Trainable
  GloVe Embedding      Embedding          Embedding
- Pooling Layer    - Pooling Layer    - Pooling Layer
- Deep Stack of    - Deep Stack of    - Shallow Stack of
  Dense Layers       Dense Layers       Dense Layers
     |                  |                  |
     +------------------+------------------+
     |
     V
[Concatenate All Three Outputs]
     |
     V
[Final Dense Layer (ReLU)]
     |
     V
[Output Layer (Sigmoid) -> 20 Probabilities]
```

  * **Network 1 (The Generalist):** This stream uses the pre-trained GloVe embedding layer, which was set to be non-trainable. Its job is to provide a rich, general understanding of the text based on what it learned from the vast GloVe corpus.
  * **Networks 2 & 3 (The Specialists):** These two streams use their own trainable embedding layers. They start with random vectors and learn word meanings that are specifically optimized for predicting occupations from bios. I used two separate specialist networks with different depths in their dense layers to allow the model to capture features at different levels of abstraction.
  * **The Merger:** The outputs from all three streams are concatenated into a single, larger vector. This vector now contains features from the generalist *and* the specialists. This combined feature set is then passed through a final set of dense layers to produce the 20 output probabilities, one for each potential occupation.

## Finding the Sweet Spot: Training and Thresholding

The model was compiled with `binary_crossentropy` loss and the `adam` optimizer, which are standard choices for a multi-label setup.

After training, the model outputs a probability for each of the 20 occupations. The final step is to decide which probabilities are high enough to count as a "yes". We do this by setting a **threshold ($$\theta$$)**. If the probability for 'Actor' is greater than $$\theta$$, we predict 'Actor'.

Finding the right $$\theta$$ is critical.

  * A **low threshold** increases recall (you find more true occupations) but hurts precision (you get more false positives).
  * A **high threshold** increases precision but hurts recall.

To find the optimal balance, I calculated the F1 score (the harmonic mean of precision and recall) for the test set using $$\theta$$ values from 0.10 to 0.80.

The results were clear. The F1 score peaked when the threshold was **0.58**. At this value, the model achieved its best performance:

  * **Optimal Theta ($$\theta$$): 0.58**
  * **F1 Score: 0.793**
  * **Precision: 0.819**
  * **Recall: 0.768**

## Conclusion and Key Takeaways

This project was a fantastic journey into solving a non-trivial NLP problem. By moving beyond a simple, single-pipeline model, I was able to significantly improve performance.

The key takeaway is that **combining different feature extraction approaches can be incredibly powerful.** The multi-input network allowed the model to leverage both the broad, general knowledge from pre-trained GloVe embeddings and the fine-tuned, task-specific knowledge from its own trainable embeddings. This hybrid approach is what ultimately pushed the F1 score to a strong 0.793.
