---
permalink: /
title: "Academic Pages is a ready-to-fork GitHub Pages template for academic personal websites"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

-----

### **Home / About Me**

**Saad Lahlali**
PhD Candidate in 3D Computer Vision & Perception

-----

Hello\! I'm Saad, a Ph.D. candidate at Université Paris-Saclay and CEA-List, nearing the completion of my doctoral journey in 3D Perception for Autonomous Driving.

My deep-rooted passion for the applied sciences is what drives me. I find it incredibly satisfying to delve into the intricacies of the world around us, and I am particularly captivated by computer vision. It's a field that allows us to replicate and understand how we as humans perceive our environment, all through the elegance of algorithms.

My research focuses on making 3D perception models for autonomous systems more robust, reliable, and scalable. The core challenge in this field is the reliance on vast amounts of meticulously hand-labeled 3D data, which is both time-consuming and expensive to create. To tackle this, I specialize in developing novel auto-labeling and weakly-supervised learning methods. These models are designed to learn from more accessible and cheaper data, like 2D images, drastically reducing the need for extensive human annotation while maintaining high performance.

I am always on the lookout for fresh, creative ways to tackle complex problems. Staying in the loop with the latest and greatest in the field is a priority for me, ensuring I'm always learning and pushing the boundaries of what's possible.

-----

### **Research & Publications**

My research aims to bridge the gap between 2D and 3D perception, enabling robust 3D understanding from limited or weakly-labeled data. Below are my contributions to the field.

-----

#### **Cross-Modal Distillation for 2D/3D Multi-Object Discovery from 2D Motion**

  * **Saad Lahlali\***, Sandra Kara\*, Hejer Ammar, Florian Chabot, Nicolas Granger, Hervé Le Borgne, Quoc-Cuong Pham. (\*Equal contribution)
  * [cite\_start]*Conference on Computer Vision and Pattern Recognition (CVPR), 2025.* [cite: 3422]

[cite\_start]**Summary:** Object discovery in 3D data has been underexplored and often relies on challenging 3D motion cues[cite: 44]. This paper introduces a novel framework that leverages flexible and generalizable 2D motion cues for 3D object discovery. [cite\_start]We present DIOD-3D, the first baseline for this task, and **xMOD**, a cross-modal teacher-student training framework that integrates 2D and 3D data to reduce bias and improve robustness[cite: 46, 47, 48]. [cite\_start]Our approach yields substantial performance gains, improving the F1 score by +8.7 to +15.1 points over the state-of-the-art on datasets like KITTI and Waymo[cite: 52].

-----

#### **ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only**

  * **Saad Lahlali**, Nicolas Granger, Hervé Le Borgne, Quoc-Cuong Pham.
  * [cite\_start]*Winter Conference on Applications of Computer Vision (WACV), 2025.* [cite: 1702, 3421]

[cite\_start]**Summary:** The high cost of 3D annotation is a major bottleneck for training 3D object detectors[cite: 1284]. [cite\_start]This work proposes ALPI, a weakly supervised 3D annotator that requires only 2D bounding box annotations and class size priors[cite: 1285]. [cite\_start]To overcome the ambiguity of projecting 2D boxes back into 3D space, we introduce a novel and effective solution: injecting 3D "proxy objects" with known annotations into the training data[cite: 1287]. This allows the model to learn reliable 3D features. [cite\_start]ALPI achieves performance close to fully supervised methods and is the first of its kind to be demonstrated on the challenging nuScenes dataset[cite: 1291, 1292].

-----

#### **MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection**

  * *Anonymous Submission to WACV 2026.*

[cite\_start]**Summary:** A significant challenge in weakly supervised 3D detection is *projection ambiguity*, where a single 2D box can correspond to many possible 3D poses[cite: 1801]. [cite\_start]This paper proposes MVAT, a framework that leverages the natural multi-view data captured by a moving vehicle over time to resolve this ambiguity[cite: 1803]. [cite\_start]Using a Teacher-Student distillation paradigm, the Teacher model learns from temporally aggregated point clouds of static objects to generate high-quality pseudo-labels[cite: 1805, 1806]. The Student model then learns to predict these labels from a single viewpoint, making it effective for both static and moving objects. [cite\_start]MVAT achieves state-of-the-art performance, significantly narrowing the gap with fully supervised methods without using any 3D annotations[cite: 1808].

-----

#### **Efficient Class-Incremental Segmentation Learning via Expanding Visual Transformers (TILES)**

  * *Anonymous Submission to TMLR.*

[cite\_start]**Summary:** This work addresses the challenge of *class-incremental semantic segmentation*, where a model must learn new object categories without forgetting old ones, a critical capability for real-world robotics and autonomous systems[cite: 2899, 2901]. [cite\_start]We focus on model efficiency, particularly for scenarios with severe memory constraints[cite: 2892]. [cite\_start]We propose **TILES**, a novel approach based on an expanding Vision Transformer (ViT) architecture that learns new tasks by adding small, specialized branches[cite: 2893]. [cite\_start]This method avoids "catastrophic forgetting" and outperforms previous methods while using up to 14 times fewer parameters[cite: 2896].

-----

### **Experience & Education**

#### **Experience**

  * **PhD Candidate in Perception for Autonomous Driving** | CEA-List

      * [cite\_start]*Dec 2022 - Present* [cite: 1717]
      * [cite\_start]Developing novel architectures for robust multi-modal object detection and segmentation with limited labeled data using self-supervised and weakly supervised approaches. [cite: 3388]

  * **Computer Vision Research Intern: Incremental Learning** | CEA-List

      * [cite\_start]*Apr 2022 - Oct 2022* [cite: 1721, 3401]
      * [cite\_start]Developed novel incremental learning methods for semantic segmentation in resource-constrained settings using a Transformer network, achieving state-of-the-art results. [cite: 3402, 3403]

  * **Research Intern: Alzheimer Detection from Handwriting** | Institut Polytechnique de Paris

      * [cite\_start]*Jun 2021 - Sep 2021* [cite: 1727, 3407]
      * [cite\_start]Developed a robust algorithm using CNNs to detect early signs of Alzheimer's disease from handwriting samples and improved classification accuracy by 10% using an ensemble learning technique. [cite: 3408, 3409]

  * **Computer Vision Intern: Crowd Counting** | Opinaka

      * [cite\_start]*Jun 2020 - Aug 2020* [cite: 1743, 3412]
      * [cite\_start]Designed a Scale-Attention Autoencoder with Inception Modules in TensorFlow for crowd counting and optimized inference time by 50%. [cite: 3414]

#### **Education**

  * **Ph.D. in Perception for Autonomous Driving**

      * [cite\_start]Université Paris-Saclay & CEA-List, France (2022 - 2025) [cite: 1704, 1749]

  * **Engineering Cycle & MSc. in Data Science and Artificial Intelligence**

      * [cite\_start]Institut Polytechnique de Paris (Télécom Paris & École Polytechnique), France (2019 - 2022) [cite: 1751, 3390, 3393]

  * **Preparatory Classes for French Grandes Ecoles**

      * [cite\_start]Mathematics & Physics (2016 - 2019) [cite: 1755, 3396]

-----

### **Contact**

I am always open to discussing research, exploring new ideas, or connecting with fellow professionals in the field. Please feel free to reach out.

  * [cite\_start]**Email:** [saad7lahlali@gmail.com](mailto:saad7lahlali@gmail.com) [cite: 1688, 3382]
  * [cite\_start]**LinkedIn:** [linkedin.com/in/saad-lahlali](https://www.google.com/search?q=https://www.linkedin.com/in/saad-lahlali) [cite: 1689, 3382]
  * [cite\_start]**GitHub:** [github.com/saad2050lahlali](https://www.google.com/search?q=https://github.com/saad2050lahlali) [cite: 1691, 3382]