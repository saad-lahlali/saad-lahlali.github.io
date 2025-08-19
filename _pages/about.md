---
permalink: /
title: "Saad Lahlali personal website"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

-----
### **Home / About Me**

**Saad Lahlali**
PhD Candidate in 3D Computer Vision & Perception

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
  * *Conference on Computer Vision and Pattern Recognition (CVPR), 2025.* **Summary:** Object discovery in 3D data has been underexplored and often relies on challenging 3D motion cues. This paper introduces a novel framework that leverages flexible and generalizable 2D motion cues for 3D object discovery. We present DIOD-3D, the first baseline for this task, and **xMOD**, a cross-modal teacher-student training framework that integrates 2D and 3D data to reduce bias and improve robustness. Our approach yields substantial performance gains, improving the F1 score by +8.7 to +15.1 points over the state-of-the-art on datasets like KITTI and Waymo.

-----

#### **ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only**

  * **Saad Lahlali**, Nicolas Granger, Hervé Le Borgne, Quoc-Cuong Pham.
  * *Winter Conference on Applications of Computer Vision (WACV), 2025.* **Summary:** The high cost of 3D annotation is a major bottleneck for training 3D object detectors. This work proposes ALPI, a weakly supervised 3D annotator that requires only 2D bounding box annotations and class size priors. To overcome the ambiguity of projecting 2D boxes back into 3D space, we introduce a novel and effective solution: injecting 3D "proxy objects" with known annotations into the training data. This allows the model to learn reliable 3D features. ALPI achieves performance close to fully supervised methods and is the first of its kind to be demonstrated on the challenging nuScenes dataset.

-----

#### **MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection**

  * *Anonymous Submission to WACV 2026.*

**Summary:** A significant challenge in weakly supervised 3D detection is *projection ambiguity*, where a single 2D box can correspond to many possible 3D poses. This paper proposes MVAT, a framework that leverages the natural multi-view data captured by a moving vehicle over time to resolve this ambiguity. Using a Teacher-Student distillation paradigm, the Teacher model learns from temporally aggregated point clouds of static objects to generate high-quality pseudo-labels. The Student model then learns to predict these labels from a single viewpoint, making it effective for both static and moving objects. MVAT achieves state-of-the-art performance, significantly narrowing the gap with fully supervised methods without using any 3D annotations.

-----

#### **Efficient Class-Incremental Segmentation Learning via Expanding Visual Transformers (TILES)**

  * *Anonymous Submission to TMLR.*

**Summary:** This work addresses the challenge of *class-incremental semantic segmentation*, where a model must learn new object categories without forgetting old ones, a critical capability for real-world robotics and autonomous systems. We focus on model efficiency, particularly for scenarios with severe memory constraints. We propose **TILES**, a novel approach based on an expanding Vision Transformer (ViT) architecture that learns new tasks by adding small, specialized branches. This method avoids "catastrophic forgetting" and outperforms previous methods while using up to 14 times fewer parameters.

-----

### **Experience & Education**

#### **Experience**

  * **PhD Candidate in Perception for Autonomous Driving** | CEA-List

      * *Dec 2022 - Present* * Developing novel architectures for robust multi-modal object detection and segmentation with limited labeled data using self-supervised and weakly supervised approaches. 

  * **Computer Vision Research Intern: Incremental Learning** | CEA-List

      * *Apr 2022 - Oct 2022* * Developed novel incremental learning methods for semantic segmentation in resource-constrained settings using a Transformer network, achieving state-of-the-art results. 

  * **Research Intern: Alzheimer Detection from Handwriting** | Institut Polytechnique de Paris

      * *Jun 2021 - Sep 2021* * Developed a robust algorithm using CNNs to detect early signs of Alzheimer's disease from handwriting samples and improved classification accuracy by 10% using an ensemble learning technique. 

  * **Computer Vision Intern: Crowd Counting** | Opinaka

      * *Jun 2020 - Aug 2020* * Designed a Scale-Attention Autoencoder with Inception Modules in TensorFlow for crowd counting and optimized inference time by 50%. 

#### **Education**

  * **Ph.D. in Perception for Autonomous Driving**

      * Université Paris-Saclay & CEA-List, France (2022 - 2025) 

  * **Engineering Cycle & MSc. in Data Science and Artificial Intelligence**

      * Institut Polytechnique de Paris (Télécom Paris & École Polytechnique), France (2019 - 2022) 

  * **Preparatory Classes for French Grandes Ecoles**

      * Mathematics & Physics (2016 - 2019) 

-----

### **Contact**

I am always open to discussing research, exploring new ideas, or connecting with fellow professionals in the field. Please feel free to reach out.

  * **Email:** [saad7lahlali@gmail.com](mailto:saad7lahlali@gmail.com) 
  * **LinkedIn:** [linkedin.com/in/saad-lahlali](https://www.google.com/search?q=https://www.linkedin.com/in/saad-lahlali) 
  * **GitHub:** [github.com/saad2050l](https://www.google.com/search?q=https://github.com/saad2050l) 