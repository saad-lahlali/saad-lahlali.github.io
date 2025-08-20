---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* **Ph.D. in Perception for Autonomous Driving**, Paris-Saclay University & CEA LIST, 2022-2025 (expected)
* **Engineering Cycle & M.Sc. in Data Science and Artificial Intelligence**, Institut Polytechnique de Paris, 2019-2022
* **Preparatory Classes (Mathematics and Physics)**, 2016-2019

Work experience
======
* **Apr. 2022 - Oct. 2022: Computer Vision Research Intern**
    * CEA LIST, Palaiseau, France
    * Duties included: Developed novel incremental learning methods for semantic segmentation in resource-constrained settings using a Transformer network.
    * Outcome: Achieved state-of-the-art results.

* **Jun. 2021 - Sep. 2021: Research Intern**
    * Institut Polytechnique de Paris, Palaiseau, France
    * Duties included: Developed a robust algorithm using CNNs to detect early signs of Alzheimer's disease from handwriting samples and implemented an ensemble learning technique.

* **Jun. 2020 - Sep. 2020: Computer Vision Intern**
    * Opinaka, Montpellier, France
    * Duties included: Explored crowd counting algorithms, designed a Scale-Attention Autoencoder in TensorFlow, and optimized inference time.

* **Jan. 2020 - Jun. 2020: Intern, Facial Emotion Analysis**
    * VocaCoach, Paris, France
    * Duties included: Cleaned and augmented the FER2013 image dataset, reduced class imbalance, and developed a seven-class emotion classification network using VGG19.

Skills
======
* **Machine Learning**: Self-supervised Learning, Weakly Supervised Learning, Incremental Learning, Ensemble Learning, Transfer Learning
* **Deep Learning**:
    <!-- * **Architectures**: Transformers, Convolutional Neural Networks (CNNs), Autoencoders -->
    * **Frameworks**: Pytorch, TensorFlow, OpenMMLab
* **Computer Vision**: 3D Object Detection, Semantic Segmentation, Crowd Counting, Facial Emotion Analysis, Medical Imaging
* **Programming & Data Science**: Python, Object Oriented Programming, Statistics, Optimization, Databases, Data Augmentation

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

<!-- Playlists
======
  <ul>{% for post in site.playlists reversed %}
    {% include archive-single-playlist-cv.html  %}
  {% endfor %}</ul> -->

<!-- Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul> -->
