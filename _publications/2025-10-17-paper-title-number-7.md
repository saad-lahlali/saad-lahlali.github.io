---
title: "MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection"
collection: publications
category: conferences
permalink: /publication/MVAT
excerpt: '<img src="/images/fig_1_new_page-0001.jpg" alt="A diagram illustrating the MVAT framework" style="float: left; margin: 0em 1em 0em 0em; width: 390px;">
 MVAT introduces a novel approach to weakly supervised 3D detection by tackling projection ambiguity. It presents a **Teacher-Student framework** that is the first to leverage the natural multi-view consistency from a moving ego-vehicle to resolve the inherent ambiguity of 2D annotations. The method proposes a robust technique to generate dense, high-quality 3D object representations and pseudo-labels by **aggregating sparse point clouds** over time, guided only by 2D bounding boxes. This process is further strengthened by a **multi-view 2D projection loss** that serves as a powerful supervisory signal, enforcing that a single predicted 3D box must align with all of its corresponding 2D annotations across the entire temporal sequence.'
date: 2026-06-01
venue: 'WACV'
paperurl: 'https://arxiv.org/abs/2509.07507'
codeurl:
citation: 
---


# Abstract
Annotating 3D data remains a costly bottleneck for 3D object detection, motivating the development of weakly supervised annotation methods that rely on more accessible 2D box annotations. However, relying solely on 2D boxes introduces projection ambiguities since a single 2D box can correspond to multiple valid 3D poses. Furthermore, partial object visibility under a single viewpoint setting makes accurate 3D box estimation difficult. We propose MVAT, a novel framework that leverages temporal multi-view present in sequential data to address these challenges. Our approach aggregates object-centric point clouds across time to build 3D object representations as dense and complete as possible. A Teacher-Student distillation paradigm is employed: The Teacher network learns from single viewpoints but targets are derived from temporally aggregated static objects. Then the Teacher generates high quality pseudo-labels that the Student learns to predict from a single viewpoint for both static and moving objects. The whole framework incorporates a multi-view 2D projection loss to enforce consistency between predicted 3D boxes and all available 2D annotations. Experiments on the nuScenes and Waymo Open datasets demonstrate that MVAT achieves state-of-the-art performance for weakly supervised 3D object detection, significantly narrowing the gap with fully supervised methods without requiring any 3D box annotations.


# Context and Motivation

* **The Problem with 3D Annotation:** 3D object detection, crucial for applications like self-driving cars, requires vast amounts of data. However, annotating 3D bounding boxes for objects in point clouds is extremely expensive and slow (up to 16 times slower than 2D annotation).

* **Weakly Supervised Learning as a Solution:** To reduce costs, researchers have turned to "weakly supervised" methods. These methods use cheaper and more accessible **2D bounding box annotations** from camera images to train 3D detection models.

* **The Core Challenges:** Relying only on 2D boxes creates two significant issues:
    1.  **Projection Ambiguity:** A single 2D box can correspond to many possible 3D boxes. It's hard to determine the object's true depth, size, and orientation from one 2D view.
    2.  **Partial Visibility:** Objects are often occluded or only partially visible from a single viewpoint, making it difficult to infer their complete 3D shape.

Existing methods try to solve this by using heuristics or priors about object shapes, but these are often not robust enough.

# The Proposed Method: MVAT

The key idea of MVAT is to use the **temporal multi-view data** that is naturally available in sequential recordings (e.g., from a moving car). By observing an object from multiple viewpoints over time, the model can gather more geometric information and resolve the ambiguities present in a single view.

The method uses a two-phase **Teacher-Student distillation framework**:

1.  **Phase 1: Teacher Burn-in**
    * **Data Aggregation:** For **static objects**, the system aggregates their corresponding point clouds from multiple frames. This creates a denser and more complete 3D representation of the object than any single frame could provide.
    * **Coarse 3D Labels:** From these aggregated point clouds, the system generates initial, "coarse" 3D bounding box pseudo-labels without any manual 3D annotation.
    * **Teacher Training:** A "Teacher" network is trained on **single-frame** views of these static objects, using the coarse 3D labels for supervision. It is forced to learn robust 3D geometry from sparse, partial views.
    * **Multi-View 2D Loss:** A crucial component is a loss function that projects the predicted 3D box back into *all* available 2D camera views and ensures it aligns with the 2D ground-truth boxes. This enforces geometric consistency across the sequence.

2.  **Phase 2: Teacher-Student Distillation**
    * **High-Quality Pseudo-Labeling:** The trained Teacher is then used to generate high-quality 3D pseudo-labels. For static objects, it uses the complete, aggregated point clouds (an easier task), and for **moving objects**, it uses single-frame point clouds.
    * **Student Training:** A "Student" network is trained on the entire dataset (both static and moving objects) using only **single-frame inputs**. It learns to replicate the Teacher's accurate predictions.
    * **Generalization:** This process distills the Teacher's knowledge (learned from rich, aggregated data) into the Student, creating a powerful and generalist detector that can accurately predict 3D boxes from a single frame, handling both static and moving objects, as well as occlusions.

In essence, MVAT bootstraps a powerful 3D annotator using only 2D labels by cleverly exploiting the geometric constraints provided by multiple views over time. This significantly closes the performance gap between weakly supervised and fully supervised methods.
# Citation
Lahlali, S., Fournier Montgieux, A., Granger, N., Le Borgne, H., & Pham, Q. C. (2026). MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV).