---
title: "Cross-Modal Distillation for 2D/3D Multi-Object Discovery from 2D Motion"
collection: publications
category: conferences
permalink: /publication/xMOD
excerpt: 'Object discovery, the task of localizing objects without human annotations, is well-studied in 2D but under-explored in 3D, where methods often rely on challenging 3D motion cues. This paper introduces a new framework that leverages flexible and generalizable 2D motion cues for 3D object discovery, bridging the gap between the two modalities. We present two main contributions: (i) DIOD-3D, a baseline for 3D multi-object discovery using 2D motion, which uses scene completion to handle sparse input data, and (ii) XMOD, a cross-modal, teacher-student training framework that integrates 2D and 3D data to reduce confirmation bias. The final model supports RGB-only, point cloud-only, or multi-modal inputs at inference, with a late-fusion technique to further boost performance. Our approach shows significant performance gains on synthetic (TRIP-PD) and real-world (KITTI, Waymo) datasets, with F1 score improvements ranging from +8.7 to +15.1 over the 2D state-of-the-art.'
date: 2025-06-01
venue: 'CVPR'
paperurl: https://openaccess.thecvf.com/content/CVPR2025/html/Lahlali_Cross-Modal_Distillation_for_2D3D_Multi-Object_Discovery_from_2D_Motion_CVPR_2025_paper.html
codeurl: https://github.com/CEA-LIST/xMOD/tree/main
citation: '@InProceedings{Lahlali_2025_CVPR,
    author    = {Lahlali, Saad and Kara, Sandra and Ammar, Hejer and Chabot, Florian and Granger, Nicolas and Le Borgne, Herve and Pham, Quoc-Cuong},
    title     = {Cross-Modal Distillation for 2D/3D Multi-Object Discovery from 2D Motion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {24529-24538}
}'
---

Object discovery, which refers to the process of localizing objects without human annotations, has gained significant attention in recent years. Despite the growing interest in this task for 2D images, it remains under-explored in 3D data, where it is typically restricted to localizing a single object. Our work leverages the latest advances in 2D object discovery and proposes a novel framework to bridge the gap between 2D and 3D modalities. Our primary contributions are twofold: (i) we propose DIOD-3D, the first method for multi-object discovery in 3D data, using scene completion as a supporting task to enable dense object discovery from sparse inputs; (ii) we develop xMOD, a cross-modal training framework that integrates both 2D and 3D data, using objective functions tailored to accommodate the sparse nature of 3D data. xMOD uses a teacher-student training across the two modalities to reduce confirmation bias by leveraging the domain gap. During inference, the model supports RGB-only, point cloud-only and multi-modal inputs. We validate the approach in the three settings, on synthetic photo-realistic and real-world datasets. Notably, our approach yields a substantial improvement in score compared with the state of the art by points in real-world scenarios, demonstrating the potential of cross-modal learning in enhancing object discovery systems without additional annotations.
