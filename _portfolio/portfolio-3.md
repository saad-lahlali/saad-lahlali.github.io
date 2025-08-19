---
title: "From 'Paris' to 'Paris_(mythology)': Building a Wikipedia Entity Disambiguator from Scratch"
excerpt: "How does a computer know if 'Paris' refers to the city of love or the hero of Greek mythology? The answer is context. This post is a step-by-step guide on building an Entity Disambiguation system from scratch in Python. Dive in to see how we can teach a machine to read between the lines by extracting meaningful context with NLTK, leveraging a knowledge base like YAGO, and implementing a simple yet powerful scoring model to resolve ambiguity in text."
collection: portfolio
---

-----

*A step-by-step guide to linking ambiguous names in text to a knowledge base using Python, NLTK, and a dash of creative similarity scoring.*

Words are wonderfully ambiguous. When you read the name "Paris," are you thinking of the capital of France, the hero of Greek mythology, or perhaps Paris Hilton? Humans use context to figure this out instantly. For a machine, however, this is a classic challenge known as **Entity Disambiguation** or **Entity Linking**.

The goal is to take a mention of an entity in a text (like "Paris") and link it to its unique, unambiguous entry in a knowledge base (like YAGO or Wikidata). Given a snippet from a Wikipedia article:

`<Paris_17>`
`Paris is a figure in the Greek mythology.`

The mission is to programmatically determine that `<Paris_17>` is, in fact, `<Paris_(mythology)>`.

In this post, I'll walk you through how I built a system to do just that, using a simplified Wikipedia dataset and the YAGO knowledge base. Let's dive in.

### The Challenge: A Sea of Ambiguity

The project's foundation consists of two key datasets:

1.  **`wikipedia-ambiguous.txt`**: A collection of Simple Wikipedia articles where the title is ambiguous (e.g., `<Babilonia_0>`).
2.  **YAGO Knowledge Base (`.tsv` file)**: A structured database of facts. For example, it knows that `<Babilonia>` is a film, its director was Jorge Salvador, and it was released in "1987".

Our task is to build the `disambiguate()` function that takes the ambiguous label ("Babilonia") and the article's content and returns the correct, unique YAGO entity (`<Babilonia>`).

### The Game Plan: A Four-Step Approach

My strategy was to build a system that mimics how a human might solve this problem: by comparing contexts. If the article mentions "film," "director," and "Argentina," it's probably talking about the movie, not the ancient city.

I broke the process down into four main steps:

1.  **Candidate Generation**: Find all possible "Parises" in our knowledge base.
2.  **Context is King**: Extract the most meaningful words from the source article.
3.  **Leveraging the Knowledge Base**: Gather contextual clues for each potential candidate.
4.  **The Showdown**: Score the candidates by comparing their context with the article's context and pick the winner.

### Getting Our Hands Dirty: The Implementation

Here’s how I translated that plan into Python code using the `nltk` library.

#### **Step 1: Extracting Meaningful Context**

First, I needed a way to pull out the most important terms from the article text. Simply using all the words would be too noisy. I created a `filter_content` function that:

  * Uses NLTK for **Part-of-Speech (POS) tagging** to identify nouns, adjectives, etc.
  * Parses the tagged text to extract **Noun Phrases** (like "Greek mythology") and any capitalized or numeric words, which are often important entities or dates.
  * Removes common "stopwords" (`the`, `a`, `in`) to reduce noise.
  * Removes the original ambiguous words themselves (e.g., removes "Paris" when analyzing the Paris article) to ensure we're matching based on surrounding context, not the term itself.

This leaves us with a clean list of context words for the article.

#### **Step 2: Finding the Candidates**

With the label (e.g., "Paris") in hand, the code scans the YAGO knowledge base to find all entities whose names contain that label. This gives us a list of potential candidates to evaluate, such as `<Paris_(mythology)>`, `<Paris,_Texas>`, and `<Paris_(The_Cure_album)>`.

#### **Step 3: The Scoring Mechanism (Our "Cosine Similarity")**

This is the core of the project. For each candidate, we need a way to score how well it matches the article's context.

I initially considered a standard cosine similarity, but a simple **word overlap score** proved to be quite effective. My `cos_sim` function calculates the number of common words between two sets of text. The logic is straightforward: the more shared vocabulary between the article's context and the knowledge base's context for a candidate, the higher the score.

The final score for a candidate entity is the sum of two overlap scores:

1.  Overlap between the *article's context* and the *candidate's knowledge base context*.
2.  Overlap between the *article's context* and the *candidate's own name/label*. (This helps in cases like `<Hawarden,_New_Zealand>`, where "New Zealand" is a strong clue).

Mathematically, the score for a candidate $C$ is:
$$Score(C) = |W_{article} \cap W_{C, KB}| + |W_{article} \cap W_{C, label}|$$
Where $W\_{article}$ is the set of context words from the article, $W\_{C, KB}$ is the context from the knowledge base for candidate C, and $W\_{C, label}$ is the set of words in the candidate's name.

The candidate with the highest score is our winner\!

### Measuring Success: Did It Work?

To evaluate the system, I used the F-beta score, specifically the **F0.5 score**, which prioritizes precision over recall. This means it’s more important to be correct when we provide an answer than it is to provide an answer for every single article.

After running the disambiguation on the development dataset, the results were:

  * **Precision**: 72.47%
  * **Recall**: 66.62%
  * **F0.5 Score**: 71.22

This is a very respectable result for a system built from scratch without complex machine learning models\!

The distribution of similarity scores for the chosen entities provides some interesting insight.

As you can see, most correct matches were made with a relatively low similarity score (1-4 overlapping words). This shows that even a small amount of shared, specific context can be a very powerful signal.

### Dealing with Uncertainty: The Entropy Score

Sometimes, the model is faced with several candidates that have very similar scores. In these cases, it's hard to be confident in the top choice. To measure this uncertainty, I calculated the **entropy** of the final scores for each set of candidates.

  * **Low Entropy**: One candidate's score is much higher than the others. We can be confident in this choice.
  * **High Entropy**: Several candidates have similarly high scores, indicating ambiguity.

The large spike at an entropy of 30 represents cases where the model found no suitable candidates and chose not to provide an answer. This is a good thing—it's better to abstain than to guess wildly. For the rest, the entropy is generally low, confirming that the model was usually quite certain of its choice.

### Final Thoughts and What's Next

This project demonstrates that a robust entity disambiguation system can be built with fundamental NLP techniques and a solid, context-based approach. By intelligently extracting context and using a simple but effective overlap score, we were able to successfully link ambiguous entities to their correct entries in a knowledge base.

Of course, there are many ways to improve it:

  * **Better Scoring**: Using TF-IDF to weigh important words more heavily.
  * **Semantic Similarity**: Using word embeddings (like Word2Vec or GloVe) to understand that "king" and "monarch" are related, even if they are not the same word.
  * **Graph-Based Methods**: Leveraging the connections between entities within the YAGO graph to inform the ranking.

-----
