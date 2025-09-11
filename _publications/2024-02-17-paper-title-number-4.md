---
title: "Cross-Modal Distillation for 2D/3D Multi-Object Discovery from 2D Motion"
collection: publications
category: conferences
permalink: /publication/xMOD
excerpt: '<div align="center">  <img src="/images/visu_xmod.jpg" alt="A diagram illustrating the XMOD framework" style="width: 100%; margin-bottom: 1.5em;"> </div> While object discovery in 2D images has thrived using motion cues, its 3D counterpart has been hindered by the sparsity of LiDAR data and unreliable 3D motion. Our work bridges this modality gap by introducing a new paradigm that leverages mature 2D motion cues to discover multiple objects directly in 3D point clouds. We first propose DIOD-3D, a novel baseline that learns scene completion to find dense objects in sparse data. Building on this, our core contribution is xMOD, a cross-modal distillation framework where 2D and 3D models act as teachers for each other. The 2D student learns robust geometry from its 3D teacher, while the 3D student learns rich context from its 2D teacher, reducing confirmation bias and exploiting the unique strengths of each sensor. '
date: 2025-06-01
venue: 'CVPR'
paperurl: https://openaccess.thecvf.com/content/CVPR2025/html/Lahlali_Cross-Modal_Distillation_for_2D3D_Multi-Object_Discovery_from_2D_Motion_CVPR_2025_paper.html
codeurl: https://github.com/CEA-LIST/xMOD/tree/main
citation: 
---

# Abstract
Object discovery, which refers to the process of localizing objects without human annotations, has gained significant attention in recent years. Despite the growing interest in this task for 2D images, it remains under-explored in 3D data, where it is typically restricted to localizing a single object. Our work leverages the latest advances in 2D object discovery and proposes a novel framework to bridge the gap between 2D and 3D modalities. Our primary contributions are twofold: (i) we propose DIOD-3D, the first method for multi-object discovery in 3D data, using scene completion as a supporting task to enable dense object discovery from sparse inputs; (ii) we develop xMOD, a cross-modal training framework that integrates both 2D and 3D data, using objective functions tailored to accommodate the sparse nature of 3D data. xMOD uses a teacher-student training across the two modalities to reduce confirmation bias by leveraging the domain gap. During inference, the model supports RGB-only, point cloud-only and multi-modal inputs. We validate the approach in the three settings, on synthetic photo-realistic and real-world datasets. Notably, our approach yields a substantial improvement in score compared with the state of the art by points in real-world scenarios, demonstrating the potential of cross-modal learning in enhancing object discovery systems without additional annotations.

# Context and Motivation

Object discovery, the task of finding and localizing objects without any human-provided labels, is a well-explored area for 2D images and videos. Many successful 2D methods rely on **motion cues** (like optical flow) to automatically identify objects, based on the simple idea that things that move together often form a single object. 🚗

However, in the 3D world, this task is much less developed. Existing 3D methods typically use 3D motion (scene flow), but this has several drawbacks:
* **Sparsity:** LiDAR data is often sparse, especially for distant objects, making 3D motion difficult to calculate reliably.
* **Complexity:** Processing 3D motion often requires complex pre-processing steps.
* **Limited Generality:** Models trained on 3D motion cues often don't generalize well to new environments.

The key motivation of this paper is to **bridge the gap between 2D and 3D object discovery**. The authors argue that the more robust and flexible motion cues from 2D videos can be effectively used to find objects in 3D point clouds, overcoming the limitations of relying on 3D motion alone.

---
# Ideas of the Proposed Method

The paper introduces a novel framework with two main components:

## 1. DIOD-3D: A New Baseline for 3D Object Discovery

This is the foundational method for finding multiple objects in 3D data using 2D motion.
* **2D Motion Guidance:** Instead of calculating motion in the 3D point cloud, the system projects the 3D data into a 2D "front-view" image. It then uses motion masks generated from the corresponding standard 2D camera image to guide the discovery process. This leverages the strengths of advanced 2D motion estimation.
* **Scene Completion Task:** A major challenge with LiDAR data is its sparsity (missing points). To combat this, the model is given an auxiliary task: **scene completion**. It learns to fill in the gaps in the projected 3D data. This forces the model to develop a better understanding of the scene's structure, which in turn helps it identify more complete and accurate objects.

## 2. XMOD: Cross-Modal Distillation Framework  Cross-Modal Distillation Framework

This is the core innovation, designed to make the system more robust by forcing the 2D and 3D modalities to learn from each other.
* **Teacher-Student Paradigm:** The framework sets up two parallel systems, one for 2D images and one for 3D point clouds. Each system has a "student" model and a "teacher" model.
* **Cross-Modal Learning:** The key idea is that the **teacher from one modality provides pseudo-labels to supervise the student of the *other* modality**.
    * The 2D teacher (which is good at understanding textures and colors) helps train the 3D student, improving its performance in situations where 3D data is weak (e.g., objects with low reflectivity).
    * The 3D teacher (which understands geometry and structure) helps train the 2D student, making it more robust in challenging lighting conditions (e.g., night scenes) where cameras struggle.
* **Benefits:** This cross-modal training reduces the risk of "confirmation bias" where a model only learns from its own mistakes. By getting feedback from a different data source, both models become more accurate and resilient.
* **Late Fusion:** During inference, if both camera and LiDAR data are available, a late-fusion technique combines the predictions from both the 2D and 3D models. It keeps only the object predictions that are consistent across both modalities, filtering out noise and improving final accuracy.

In essence, the paper proposes a synergistic approach where the maturity of 2D object discovery is used to kickstart and enhance 3D object discovery, and then a cross-training framework allows both domains to benefit from each other's strengths.

# Citation
Lahlali, S., Kara, S., Ammar, H., Chabot, F., Granger, N., Le Borgne, H., & Pham, Q.-C. (2025). Cross-Modal Distillation for 2D/3D Multi-Object Discovery from 2D Motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 24529–24538).