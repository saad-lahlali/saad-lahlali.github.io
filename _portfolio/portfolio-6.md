---
title: "How I Built a Simple AI to Understand Wikipedia’s First Sentence"
excerpt: "How can a machine learn that Leicester is a city just by reading a sentence? I built a simple AI that does just that. Discover how fundamental NLP techniques, not complex deep learning, can be used to extract key information from Wikipedia with surprising accuracy."
collection: portfolio
---

-----
## A step-by-step guide to extracting entity types (like ‘city’ from ‘Leicester is a city’) using Python, NLTK, and a clever set of rules.

Wikipedia is the largest encyclopedia ever created. It’s a vast treasure trove of human knowledge, but it's written for humans, not machines. A person can read "Leicester is a city in England" and instantly understand what Leicester is. But how can we teach a computer to do the same thing at scale?

The answer often lies in the very first sentence.

In this post, I'll walk you through a project I worked on to build a simple but surprisingly effective program that automatically extracts the "type" of a thing from its Wikipedia page. The goal was to read an article's title and first sentence, like:

  * **Title:** Leicester
  * **Content:** Leicester is a small city in England.

And produce a clean, machine-readable fact: `Leicester TAB city`.

I’ll break down the method I used, which relies on some fundamental concepts from Natural Language Processing (NLP) to achieve an accuracy score of over 84%.

### The Core Idea: Sentences Follow Patterns

Language isn't random. Most definitional sentences follow a predictable structure. Think about it:

  * "**April** is the fourth **month** of the year..."
  * "**Acanthocephala** is a **phylum** of parasitic worms..."
  * "The **Southern Railway** was a British railway **company**..."

The pattern is often: **[Entity]** is a/an [optional adjectives] **[TYPE]**.

Our entire strategy is to teach the machine how to recognize this grammatical pattern. To do that, we need a way to see the *structure* of a sentence, not just the words.

### Step 1: Deconstructing Sentences with NLTK

The first step is to break a sentence down into its components and label each word with its part of speech (POS). Is it a noun, a verb, an adjective? This is called POS Tagging.

For this task, I used Python’s most famous NLP library, the Natural Language Toolkit (`nltk`). It makes this process incredibly simple.

Let's take the sentence, "Leicester is a beautiful English city in the UK." First, we tokenize it (split it into individual words) and then we tag it:

```python
import nltk
from nltk.tokenize import word_tokenize

sent = "Leicester is a beautiful English city"
tokens = word_tokenize(sent.lower())
pos_tags = nltk.pos_tag(tokens)

print(pos_tags)
```

This gives us a labeled list of tuples, where each word is paired with its POS tag:

```
[('leicester', 'NN'), ('is', 'VBZ'), ('a', 'DT'), ('beautiful', 'JJ'), ('english', 'JJ'), ('city', 'NN')]
```

Here’s what those tags mean:

  * `NN`: Singular Noun
  * `VBZ`: Verb, 3rd person singular present (like "is")
  * `DT`: Determiner (like "a" or "the")
  * `JJ`: Adjective

Now that the sentence has a clear grammatical structure, we can hunt for our pattern.

### Step 2: The Secret Sauce — Regular Expressions on Grammar

This is where the magic happens. We need to define a rule that says, "look for a sequence that looks like a verb, then maybe a determiner, then maybe some adjectives, and finally, one or more nouns."

Using `nltk`'s `RegexpParser`, we can define these rules using the POS tags. This is called "chunking"—we’re finding chunks of text that match our pattern. I created a list of several patterns to catch different sentence structures, but here is a key one that handled many cases:

```python
# A pattern to find a Noun Phrase (NP) that follows a verb
pattern = "NP: {<.>?(<VBZ>|<VBP>|<VBD>)<DT>?<RB>*(<JJ>+|<JJ>?)(<NN>+|<NNS>+)}"
```

That looks complicated, so let's break it down:

  * `NP:`: We're defining a chunk called a Noun Phrase.
  * `<VBZ>|<VBP>|<VBD>`: Find a verb like "is", "are", or "was".
  * `<DT>?`: Look for an optional (`?`) determiner ("a", "the").
  * `<JJ>+|<JJ>?`: Look for one or more (`+`) or zero-to-one (`?`) adjectives.
  * `<NN>+|<NNS>+`: The grand prize. Find one or more (`+`) singular (`NN`) or plural (`NNS`) nouns. This is our target\!

When we apply this pattern to our tagged sentence about Leicester, it correctly identifies "city" as the noun that fits the rule.

### Step 3: Refining the Output

Of course, it’s not always that clean. Sometimes the model would extract overly generic terms. For example, "A tablespoon is a **kind** of spoon." We want "spoon," not "kind."

To solve this, I created a small list of "unwanted" words to filter out from the results:

```python
unwanted = ['form', 'kind', 'sort', 'type', 'way', 'part', 'name', 'piece', ... ]
```

If the pattern returned any of these words, the program would ignore it and look for a better candidate. The final logic was simple: apply all the patterns to the sentence, collect all valid possibilities that aren't in the `unwanted` list, and return the one that appears earliest in the sentence.

### The Final Results

So, how did this rule-based approach perform? I ran it on a dataset of thousands of Wikipedia articles and evaluated it against a "gold standard" of correct answers.

The final evaluation, using an F-score that prioritizes precision (getting it right) over recall (finding every single one), was **84.17%**.

It performed extremely well on clear definitions:

  * **Flagstaff, Arizona:** "Flagstaff is a **city**..." -\> `city`
  * **Alec Baldwin:** "Alexander Rae 'Alec' Baldwin III is an American **actor**..." -\> `actor`
  * **8667 Fontane:** "8667 Fontane is a Main-belt **Asteroid**..." -\> `asteroid`

However, it had its limitations. The model struggled when the definition was less direct or involved more complex sentence structures. For instance:

  * **Hillary Rodham Clinton:** Expected `senator`, but the model returned `states`.
  * **Isthmus:** Expected `land`, but the model returned `strip`.
  * **Moat:** Expected `water`, but the model returned `body`.

### Conclusion: Simple Rules Can Go a Long Way

This project shows that you don't always need a massive deep learning model to solve a complex NLP problem. By observing the inherent patterns in language and translating them into a set of precise rules, we can build a highly effective information extractor.

The process was a great lesson in the power of fundamentals:

1.  **Tokenize:** Break down text into words.
2.  **POS Tag:** Assign grammatical labels.
3.  **Pattern Match:** Define rules to find what you're looking for.
4.  **Filter & Refine:** Clean up the results to get meaningful output.
