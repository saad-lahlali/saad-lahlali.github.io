---
title: "ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only"
collection: publications
category: conferences
permalink: /publication/ALPI
excerpt: '<img src="/images/visu_alpi.png" alt="A diagram illustrating the MVAT framework" style="float: left; margin: 0em 1em 0em 0em; width: 450px;">Training 3D object detectors is notoriously constrained by the cost of manual 3D annotation. While using 2D boxes is a cheap alternative, it introduces a fundamental problem: how can a model learn to predict a 3D box if it has never seen one? Our work, ALPI, resolves this paradox by introducing proxy injection: we synthetically create and inject perfectly-labeled 3D proxy objects directly into the point cloud during training. These proxies, built from simple class size priors, provide the unambiguous 3D supervision needed to bootstrap the learning process, enabling the model to correctly infer the 3D poses of real-world objects using only their 2D box projections. '
date: 2025-02-01
venue: 'WACV'
paperurl: 'https://openaccess.thecvf.com/content/WACV2025/html/Lahlali_ALPI_Auto-Labeller_with_Proxy_Injection_for_3D_Object_Detection_using_WACV_2025_paper.html'
codeurl: 'https://github.com/CEA-LIST/ALPI'  
citation: 
---

# Abstract

3D object detection is crucial for applications like autonomous driving, but training detectors requires expensive 3D annotations. This paper proposes ALPI, a weakly supervised 3D annotator that only needs 2D bounding box annotations and object size priors. To overcome the ambiguity of projecting 3D poses into 2D, the method injects 3D proxy objects with known annotations into the training data. The approach also introduces a novel depth-invariant 2D loss to better align supervision with 3D detection and uses an offline pseudo-labeling scheme to gradually improve its annotations. Experiments show the method performs as well as or better than previous works on the KITTI dataset''s Car category and achieves near fully supervised performance on more challenging classes like Pedestrian and Cyclist.

This paper introduces **ALPI (Auto-Labeller with Proxy Injection)**, a method for training 3D object detectors using only simple 2D bounding box labels from images, which are much cheaper and faster to create than full 3D annotations.

# Context and Motivation

3D object detection is essential for applications like autonomous driving and robotics. However, training these systems requires large datasets with precise 3D bounding box annotations, a process that is extremely **time-consuming and expensive**. An average of two minutes is needed to annotate a single object in 3D.

To solve this, researchers have explored "weakly supervised" methods that use cheaper, less detailed labels. Using 2D bounding boxes from camera images is a popular approach. The main challenge, however, is **projection ambiguity**: a single 2D box can correspond to many different 3D boxes at various positions and orientations, making it difficult for a model to learn the correct 3D shape and location.

Previous methods trying to solve this either still required a small amount of expensive 3D data (making them semi-weakly supervised) or used hand-crafted rules that only worked for specific object classes, like cars, and were hard to adapt to others like pedestrians or cyclists.

# Proposed Method: ALPI

ALPI is designed to overcome these limitations. It's the first method that is **multi-class** (works for cars, pedestrians, etc.) and requires **no 3D annotations at all**. The core ideas are:

1.  **Proxy Injection**: This is the key innovation to solve the projection ambiguity problem. The system creates simple, synthetic 3D "proxy" objects (cuboids) based on known size priors for each object class (e.g., the average height, width, and length of a car). These proxies, which have perfect 3D labels by construction, are then "injected" into the real-world point cloud data. The model learns fundamental 3D detection features from these perfectly labeled synthetic objects, which helps it correctly interpret the real objects that only have weak 2D labels.

2.  **Size Priors**: Instead of complex, class-specific rules, ALPI only needs basic size statistics (mean and standard deviation of height, width, length) for each new object class. This information, easily found online, makes the method highly adaptable and generalizable.

3.  **Depth-Invariant 2D Loss**: The authors introduce a new way to calculate the error between the projected 3D box and the 2D label. This novel loss function is normalized by the object's size in the image, ensuring that the training process is stable and isn't thrown off by objects being at different distances from the camera.

4.  **Iterative Pseudo-Labeling**: The system uses an offline, iterative refinement process. After an initial training phase with proxy objects, the model generates its own 3D "pseudo-labels" for the real objects in the dataset. The most confident of these predictions are then treated as if they were ground truth and are used to retrain the model in subsequent iterations. This allows the annotator to gradually improve its accuracy and learn to detect more difficult, partially occluded objects.

By combining these techniques, ALPI can generate high-quality 3D pseudo-labels that can then be used to train any standard, off-the-shelf 3D object detector, achieving performance close to models trained with expensive, fully supervised data.

# Citation
Lahlali, S., Granger, N., Le Borgne, H., & Pham, Q.-C. (2025). ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV) (pp. 2185–2194).