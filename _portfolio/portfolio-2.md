---
title: "Face recognition using Multi-Patch networks and Triplet Loss"
excerpt: "What if the key to better face recognition is to stop looking at the whole picture and start focusing on the details? Dive into our student project where we implemented a powerful system that feeds a neural network not just the full face, but also individual patches like the eyes and mouth. We'll break down the secret sauce—a technique called 'Triplet Loss' that teaches the AI to see similarities like never before. Read on to discover how this multi-patch approach dramatically boosts accuracy and what we learned along the way. <br/><img src='/images/portfolio-2/triplet_loss.png' style='display: block; margin-left: auto; margin-right: auto; width: 70%;'>"
collection: portfolio
---

Face recognition technology is everywhere, but how does it achieve such high accuracy? Is it enough to just show a neural network a picture of a face? A fascinating paper by Liu & al., "[Targeting Ultimate Accuracy Face Recognition](https://arxiv.org/ftp/arxiv/papers/1506/1506.07310.pdf)," suggests that we can do better by breaking the face down into its most important parts.

This post walks you through our journey where we set out to implement and understand the core ideas behind this powerful paper. We'll explore the two key concepts that make it work: **Multi-Patch inputs** and the **Triplet Loss** function.

### The Core Idea Part 1: The Multi-Patch Method

The first major idea is simple but powerful: instead of feeding a Convolutional Neural Network (CNN) just a single image of a face, we also provide it with specific parts of that face based on facial landmarks. Think of it like giving the network extra clues—the eyes, the mouth, and the nose, all as separate inputs alongside the full face.

Each of these patches is processed by its own CNN. The outputs from all the networks are then combined (concatenated) into a single feature vector that represents the face in much greater detail.

<p align="center">
  <img src="/images/portfolio-2/multi_patch.png">
</p>


### The Core Idea Part 2: The Triplet Loss Function

For a task like face recognition, a standard loss function isn't always the best fit. We need to teach the network to understand similarity. This is where Triplet Loss comes in.

Instead of looking at one image at a time, the network looks at three:
* **An Anchor:** The reference image.
* **A Positive:** A different image of the *same* person.
* **A Negative:** An image of a *different* person.

<p align="center">
  <img src="/images/portfolio-2/triplet_loss.png">
</p>

The goal of the Triplet Loss function is to "pull" the Anchor and Positive images closer together in the feature space while "pushing" the Anchor and Negative images further apart. More formally, it aims to ensure that the distance between the anchor and the negative is greater than the distance between the anchor and the positive, plus a certain margin. This margin helps the model handle variations like different lighting or age.

<p align="center">
  <img src="/images/portfolio-2/triplt.png">
</p>

### Our Experimental Journey

With a grasp of the theory, we set our objectives:
1.  Implement the Multi-Patch and Triplet Loss method from the paper.
2.  Study how the number of patches affects accuracy.
3.  Compare the performance of Triplet Loss against the similar Siamese Loss in this multi-patch context.

#### Crafting the Data

We used the Labeled Faces in the Wild (LFW) dataset, the same one from the original paper, which allowed for a direct comparison. This dataset provided images with varied lighting, expressions, and positions.

<p align="center">
  <img src="/images/portfolio-2/LFW.png">
</p>

Creating the patches was a multi-step process:
* First, we used the dlib library to detect 68 facial landmarks on each image. These markers helped us define the regions for our patches (eyes, nose, etc.).
* We quickly noticed a problem: the initial patches were all different sizes. CNNs require standardized input sizes.
* To solve this, we created a new method. We went through our entire training set to find the maximum height and width for each type of patch. We then used these maximum dimensions to create new, standardized patches centered on the original regions.

This standardization worked well, though it had a minor drawback: for faces near the edge of an image, the patch sometimes had to be shifted, meaning it occasionally captured areas outside the desired region.

<p align="center">
  <img src="/images/portfolio-2/landmarks.png">
</p>

<p align="center">
  <img src="/images/portfolio-2/patchs.png">
</p>


<p align="center">
  <img src="/images/portfolio-2/triplets.png">
</p>

#### The Network Architecture

Once our data triplets were generated, we built our neural network. The original paper used nine CNN layers, but we began with that and eventually reduced the number to four for our implementation. The three inputs (Anchor, Positive, Negative) are fed into three identical networks that share the same weights.




### The Results: More Patches and a Better Loss

Our experiments validated the hypotheses from the paper and our own objectives.

> The more patches are used, the greater the results. This method also performs much better with triplet loss than with siamese loss.

We tracked the accuracy of our model as we increased the number of patches, comparing both Triplet Loss and Siamese Loss.

* **Increasing Patches Boosts Accuracy:** For both loss functions, we saw a clear trend: as we added more patches, the recognition accuracy increased. The paper’s conclusion that accuracy converges after 7 patches appears to be supported by our findings.
* **Triplet Loss is the Clear Winner:** Across the board, the Triplet Loss model consistently outperformed the Siamese Loss model. This confirms that for this multi-patch architecture, Triplet Loss is the more effective choice for learning detailed facial embeddings.

### Final Takeaways

This project was a deep dive into what makes modern face recognition so effective. Our personal implementation of the method from Liu & al. led us to two main conclusions:

1.  **Context is Key:** Using multiple patches provides the network with more contextual information than a single face image, leading to better results.
2.  **The Right Tool for the Job:** Triplet Loss is a superior choice over Siamese Loss for this kind of similarity learning task, as it more robustly separates different identities.

This work was completed by Steve SUARD and [Saad LAHLALI](https://www.linkedin.com/in/saad-lahlali/) for a school project, and it gave us a practical understanding of building and testing complex deep learning architectures.

***

**Suggested Tags:** `Face Recognition`, `Deep Learning`, `Computer Vision`, `Triplet Loss`, `Student Project`