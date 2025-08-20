---
title: "ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only"
collection: publications
category: conferences
permalink: /publication/ALPI
excerpt: '3D object detection is crucial for applications like autonomous driving, but training detectors requires expensive 3D annotations. This paper proposes ALPI, a weakly supervised 3D annotator that only needs 2D bounding box annotations and object size priors. To overcome the ambiguity of projecting 3D poses into 2D, the method injects 3D proxy objects with known annotations into the training data. The approach also introduces a novel depth-invariant 2D loss to better align supervision with 3D detection and uses an offline pseudo-labeling scheme to gradually improve its annotations. Experiments show the method performs as well as or better than previous works on the KITTI dataset''s Car category and achieves near fully supervised performance on more challenging classes like Pedestrian and Cyclist.'
date: 2025-02-01
venue: 'IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)'
paperurl: 'https://openaccess.thecvf.com/content/WACV2025/html/Lahlali_ALPI_Auto-Labeller_with_Proxy_Injection_for_3D_Object_Detection_using_WACV_2025_paper.html'
codeurl: 'https://github.com/CEA-LIST/ALPI'  
citation: '@InProceedings{Lahlali_2025_WACV,
    author    = {Lahlali, Saad and Granger, Nicolas and Le Borgne, Herve and Pham, Quoc-Cuong},
    title     = {ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2185-2194}
}'
---

This research addresses the high cost of 3D annotation for object detection models by developing a method that relies only on cheaper 2D bounding box labels. A key challenge is that multiple 3D boxes can project to the same 2D box, creating an ill-posed problem.

The proposed solution, ALPI, introduces two main innovations:
* **Proxy Injection**: Synthetic 3D proxy objects, created using class size priors, are injected into the training scenes. These proxies have perfect 3D labels by construction, allowing the model to learn robust 3D features that generalize to real objects.
* **Depth-Normalized 2D Loss**: A novel loss function is proposed to make the 2D supervision consistent regardless of an object's distance from the sensor. This prevents training instability caused by varying object depths in outdoor scenes.

The method uses an iterative, offline pseudo-labeling process where the annotator is retrained with its own most confident predictions, gradually improving performance on difficult examples. The effectiveness of ALPI is demonstrated on the challenging KITTI and nuScenes datasets, showing strong performance even when using 2D labels from an automatic detector instead of manual annotations.
