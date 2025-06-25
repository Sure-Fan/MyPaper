# Paper in Detection and Segmentation
This is a collection of papers on **Object Detection** and **Semantic Segmentation**. We collected the defferent works of **Supervised**, **Weakly Supervised**, **Unsupervised** and **Open-Vocabulary** in **Object Detection** and **Semantic Segmentation**. Further more, we also provide download links to **Dataset** commonly used in Detection and Segmentation works.

## Semantic Segmentation
### Supervised Semantic Segmentation
1. [2021 ICCV] **Vision Transformers for Dense Prediction** [[paper]](https://arxiv.org/abs/2103.13413v1) [[code]](https://github.com/isl-org/DPT?tab=readme-ov-file)
2. [2021 ICCV] **Segmenter: Transformer for Semantic Segmentation** [[paper]](https://arxiv.org/abs/2105.05633) [[code]](https://github.com/rstrudel/segmenter)
3. [2022 ICLR] **Language-driven Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2201.03546) [[code]](https://github.com/isl-org/lang-seg)
4. [2025 CVPV] **Your ViT is Secretly an Image Segmentation Model** [[paper]](https://arxiv.org/pdf/2503.19108) [[code]](https://github.com/tue-mps/eomt)
### Open-Vocabulary Semantic Segmentation
#### Zero-Shot
1. [2024 CVPR] **Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Diffuse_Attend_and_Segment_Unsupervised_Zero-Shot_Segmentation_using_Stable_Diffusion_CVPR_2024_paper.pdf) [[code]](https://github.com/google/diffseg)
2. [2024 CVPR] **On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_On_the_Test-Time_Zero-Shot_Generalization_of_Vision-Language_Models_Do_We_CVPR_2024_paper.pdf) [[code]](https://github.com/MaxZanella/MTA)
3. [2024 ECCV] **OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.14183) [[code]](https://github.com/cubeyoung/OTSeg?tab=readme-ov-file)
4. [2024 ICCV] **Zero-guidance Segmentation Using Zero Segment Labels** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Rewatbowornwong_Zero-guidance_Segmentation_Using_Zero_Segment_Labels_ICCV_2023_paper.pdf) [[code]](https://github.com/nessessence/ZeroGuidanceSeg)
5. [2024 NIPS] **DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1867748a011e1425b924ec72a4066b62-Paper-Conference.pdf) [[code]](https://github.com/PaulCouairon/DiffCut)
6. [2025 ICLR] **Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model** [[paper]](https://arxiv.org/pdf/2412.18303) [[code]](https://github.com/Yushu-Li/ECALP?tab=readme-ov-file)
#### Training-Free
1. [2024 CVPR] **Clip-diy: Clip dense inference yields open-vocabulary semantic segmentation for-free** [[paper]](https://arxiv.org/pdf/2309.14289)
2. [2024 CVPR] **Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10655445&tag=1) [[code]](https://github.com/aimagelab/freeda)
3. [2024 ECCV] **Proxyclip: Proxy attention improves clip for open-vocabulary segmentation** [[paper]](https://arxiv.org/pdf/2408.04883) [[code]](https://github.com/mc-lan/ProxyCLIP?tab=readme-ov-file)
4. [2024 ECCV] **ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference** [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf) [[code]](https://github.com/mc-lan/ClearCLIP)
5. [2024 ECCV] **Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP)
6. [2024 ECCV] **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference** [[paper]](https://arxiv.org/pdf/2312.01597) [[code]](https://github.com/wangf3014/SCLIP)
7. [2024 ECCV] **Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2407.08268) [[code]](https://github.com/leaves162/CLIPtrase)
8. [2024 arXiv] **CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13836) [[code]](https://github.com/linsun449/cliper.code?tab=readme-ov-file)
9. [2025 CVPR] **LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.19777) [[code]](https://github.com/vladan-stojnic/LPOSS)
10. [2025 CVPR] **ResCLIP: Residual Attention for Training-free Dense Vision-language Inference** [[paper]](https://arxiv.org/pdf/2411.15851) [[code]](https://github.com/yvhangyang/ResCLIP?tab=readme-ov-file)
11. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Distilling_Spectral_Graph_for_Object-Context_Aware_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/MICV-yonsei/CASS)
12. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person Re-identification** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf) [[code]](https://github.com/Jinxi-Yang-WHU/Fast-GCR.git) [[note]](本文提到的很多re-ranking的技术就是对直接计算的相似度矩阵进行更新，前面公式搞了一大堆，最后就是一个特征传播。)
13. [2025 CVPR] **ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements** [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/PixFoundation/papers/Aydin_ITACLIP_Boosting_Training-Free_Semantic_Segmentation_with_Image_Text_and_Architectural_CVPRW_2025_paper.pdf) [[code]](https://github.com/m-arda-aydn/ITACLIP)
14. [2025 arXiv] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.15869) [[code]](https://github.com/SuleBai/SC-CLIP?tab=readme-ov-file)
15. [2025 arXiv] **Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.21844v1) [[code]](https://github.com/dosowiechi/MLMP?tab=readme-ov-file)
16. [2025 arXiv] **FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/abs/2504.10487) [[code]](https://github.com/yasserben/FLOSS)
#### Training
1. [2022 CVPR] **GroupViT: Semantic Segmentation Emerges from Text Supervision** [[paper]](https://arxiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT?tab=readme-ov-file)
2. [2023 CVPR] **Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning** [[paper]](https://arxiv.org/pdf/2212.04994)
3. [2023 CVPR] **Learning to Generate Text-grounded Mask for Open-world Semantic Segmentation from Only Image-Text Pairs** [[paper]](https://arxiv.org/pdf/2212.00785) [[code]](https://github.com/khanrc/tcl?tab=readme-ov-file)
4. [2023 ICML] **SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2211.14813) [[code]](https://github.com/ArrowLuo/SegCLIP?tab=readme-ov-file)
5. [2023 ICML] **Grounding Everything: Emerging Localization Properties in Vision-Language Transformers** [[paper]](https://arxiv.org/pdf/2312.00878) [[code]](https://github.com/WalBouss/GEM?tab=readme-ov-file)
6. [2023 NIPS] **Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2310.19001) [[code]](https://github.com/Ferenas/PGSeg?tab=readme-ov-file)
7. [2023 ICCV] **Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.pdf)
8. [2024 CVPR] **SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2311.15537v2) [[code]](https://github.com/xb534/SED.git)
9. [2024 CVPR] **Not All Classes Stand on Same Embeddings: Calibrating a Semantic Distance with Metric Tensor** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10658112&tag=1)
10. [2024 CVPR] **USE: Universal Segment Embeddings for Open-Vocabulary Image Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_USE_Universal_Segment_Embeddings_for_Open-Vocabulary_Image_Segmentation_CVPR_2024_paper.pdf)
11. [2024 CVPR] **CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2303.11797) [[code]](https://github.com/cvlab-kaist/CAT-Seg)
12. [2024 CVPR] **Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Emergent_Open-Vocabulary_Semantic_Segmentation_from_Off-the-shelf_Vision-Language_Models_CVPR_2024_paper.pdf) [[code]](https://github.com/letitiabanana/PnP-OVSS)
13. [2024 CVPR] **SAM-CLIP: Merging Vision Foundation Models towards Semantic and Spatial Understanding** [[paper]](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Wang_SAM-CLIP_Merging_Vision_Foundation_Models_Towards_Semantic_and_Spatial_Understanding_CVPRW_2024_paper.pdf)
14. [2024 CVPR] **Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Image-to-Image_Matching_via_Foundation_Models_A_New_Perspective_for_Open-Vocabulary_CVPR_2024_paper.pdf)
15. [2024 CVPR] **EAGLE: Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.01482) [[code]](https://github.com/MICV-yonsei/EAGLE?tab=readme-ov-file)
16. [2024 ICLR] **CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction** [[paper]](https://arxiv.org/pdf/2310.01403v2) [[code]](https://github.com/wusize/CLIPSelf?tab=readme-ov-file)
17. [2024 ICLR] **A HARD-TO-BEAT BASELINE FOR TRAINING-FREE CLIP-BASED ADAPTATION** [[paper]](https://arxiv.org/pdf/2402.04087) [[code]](https://github.com/mrflogs/ICLR24)
18. [2024 NIPS] **Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1119587863e78451f080da2a768c4935-Paper-Conference.pdf) [[code]](https://github.com/cvlab-kaist/PixelCLIP)
19. [2024 ECCV] **CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation** [[paper]](https://arxiv.org/pdf/2312.12359) [[code]](https://github.com/wysoczanska/clip_dinoiser)
20. [2024 ECCV] **In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation** [[paper]](http://arxiv.org/abs/2408.04961) [[code]](https://github.com/dahyun-kang/lavg)
21. [2024 arXiv] **DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment** [[paper]](https://arxiv.org/pdf/2412.16334)
22. [2025 CVPR] **Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.21780)
23. [2025 CVPR] **Your ViT is Secretly an Image Segmentation Model** [[paper]](https://arxiv.org/pdf/2503.19108) [[code]](https://github.com/tue-mps/eomt)
24. [2025 CVPR] **Exploring Simple Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lai_Exploring_Simple_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf)
25. [2025 CVPR] **Dual Semantic Guidance for Open Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Dual_Semantic_Guidance_for_Open_Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf)
26. [2025 CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
27. [2025 CVPR] **DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DeCLIP_Decoupled_Learning_for_Open-Vocabulary_Dense_Perception_CVPR_2025_paper.pdf) [[code]](https://github.com/xiaomoguhz/DeCLIP)
28. [2025 ICLR] **Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion** [[paper]](https://arxiv.org/pdf/2502.04263) [[code]](https://github.com/miccunifi/Cross-the-Gap?tab=readme-ov-file)
### Weakly Supervised Semantic Segmentation
1. [2024 CVPR] **Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2406.11189v1) [[code]](https://github.com/zbf1991/WeCLIP)
2. [CVPR 2024] **CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13147v1) [[code]](https://github.com/ZiqinZhou66/ZegCLIP?tab=readme-ov-file)
3. [2024 CVPR] **DuPL: Dual Student with Trustworthy Progressive Learning for Robust Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.11184) [[code]](https://github.com/Wu0409/DuPL)
4. [2024 CVPR] **Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation** [[paper]](https:https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_Hunting_Attributes_Context_Prototype-Aware_Learning_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/Barrett-python/CPAL)
5. [2024 CVPR] **Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_the_Generalization_of_Segmentation_Foundation_Model_under_Distribution_Shift_CVPR_2024_paper.pdf)
6. [2024 CVPR] **Official code for Class Tokens Infusion for Weakly Supervised Semantic Segmentation** [[paper]](Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper) [[code]](https://github.com/yoon307/CTI)
7. [2025 CVPR] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
### Unsupervised Semantic Segmentation
1. [2021 ICCV] **Emerging Properties in Self-Supervised Vision Transformers** [[paper]](http://openaccess.thecvf.com//content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [[code]](https://github.com/facebookresearch/dino) [[note]](https://blog.csdn.net/YoooooL_/article/details/129234966)
2. [2022 CVPR] **Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization** [[paper]](https://arxiv.org/abs/2205.07839) [[code]](https://github.com/lukemelas/deep-spectral-segmentation?tab=readme-ov-file)
3. [2022 CVPR] **Freesolo: Learning to segment objects without annotations** [[paper]](https://arxiv.org/pdf/2202.12181) [[code]](https://github.com/NVlabs/FreeSOLO)
4. [2022 ECCV] **Extract Free Dense Labels from CLIP** [[paper]](https://arxiv.org/pdf/2112.01071) [[code]](https://github.com/chongzhou96/MaskCLIP?tab=readme-ov-file) [[note]](https://www.cnblogs.com/lipoicyclic/p/16967704.html)
5. [2023 CVPR] **ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation** [[paper]](https://arxiv.org/abs/2212.03588) [[code]](https://github.com/ZiqinZhou66/ZegCLIP?tab=readme-ov-file)
6. [2024 CVPR] **Guided Slot Attention for Unsupervised Video Object Segmentation** [[paper]](https://arxiv.org/pdf/2303.08314v3) [[code]](https://github.com/Hydragon516/GSANet)
7. [2024 CVPR] **ReCLIP++:Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2408.06747) [[code]](https://github.com/dogehhh/ReCLIP)
8. [2024 ECCV] **Unsupervised Dense Prediction using Differentiable Normalized Cuts** [[paper]](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05675.pdf)
9. [2024 NIPS] **PaintSeg: Training-free Segmentation via Painting** [[paper]](https://arxiv.org/abs/2305.19406)
10. [2024 CVPR] **CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers** [[paper]](https://arxiv.org/pdf/2403.07700) [[code]](https://github.com/shahaf-arica/CuVLER?tab=readme-ov-file)

## Object Detection
### Supervised Camouflaged Object Detection
1. [2020 CVPR] **Camouflaged Object Detection** [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.pdf) [[code]](https://github.com/DengPingFan/SINet)
2. [2023 AAAI] **High-resolution Iterative Feedback Network for Camouflaged Object Detection** [[paper]](https://arxiv.org/pdf/2203.11624) [[code]](https://github.com/HUuxiaobin/HitNet?tab=readme-ov-file)
3. [2024 arXiv] **PlantCamo: Plant Camouflage Detection** [[paper]](https://arxiv.org/pdf/2410.17598v1) [[code]](https://github.com/yjybuaa/PlantCamo)
### Open-Vocabulary Object Detection
1. [2022 ICLR] **Open-vocabulary Object Detection via Vision and Language Knowledge Distillation** [[paper]](https://arxiv.org/pdf/2104.13921) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
2. [2022 CVPR] **Grounded Language-Image Pre-training** [[paper]](https://arxiv.org/pdf/2112.03857) [[code]](https://github.com/microsoft/GLIP)
3. [2024 CVPR] **SHiNe: Semantic Hierarchy Nexus for Open-vocabulary Object Detection** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_SHiNe_Semantic_Hierarchy_Nexus_for_Open-vocabulary_Object_Detection_CVPR_2024_paper.pdf)
### Weakly Supervised Object Detection
1. [2023 ICCV] **Spatial-Aware Token for Weakly Supervised Object Localization** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf) [[code]](https://github.com/wpy1999/SAT?tab=readme-ov-file)
2. [2023 CVPR] **Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf) [[code]](https://github.com/xulianuwa/MMCST)
### Unsupervised Object Detection
1. [2022 CVPR] **Self-supervised transformers for unsupervised object discovery using normalized cut** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.pdf) [[code]](https://www.m-psi.fr/Papers/TokenCut2022/)
2. [2023 CVPR] **Unsupervised Object Localization: Observing the Background to Discover Objects** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Simeoni_Unsupervised_Object_Localization_Observing_the_Background_To_Discover_Objects_CVPR_2023_paper.pdf) [[code]](https://github.com/valeoai/FOUND)
3. [2023 CVPR] **Weakly-supervised Contrastive Learning for Unsupervised Object Discovery** [[paper]](https://arxiv.org/pdf/2307.03376) [[code]](https://github.com/npucvr/WSCUOD)
4. [2024 CVPR] **DIOD: Self-Distillation Meets Object Discovery** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Kara_DIOD_Self-Distillation_Meets_Object_Discovery_CVPR_2024_paper.pdf) [[code]](https://github.com/CEA-LIST/DIOD)
5. [2023 ICCV] **MOST: multiple object local- ization with self-supervised transformers for object discovery** [[paper]]()
6. [2023 ICCV] **SEMPART: Self-supervised Multi-resolution Partitioning of Image Semantics** [[paper]](https://arxiv.org/pdf/2309.10972)
7. [2023 ICCV] **Box-based Refinement for Weakly Supervised and Unsupervised Localization Tasks** [[paper]](https://arxiv.org/pdf/2309.03874) [[code]](https://github.com/eyalgomel/box-based-refinement)
8. [2024 CVPR] **CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers** [[paper]](https://arxiv.org/pdf/2403.07700) [[code]](https://github.com/shahaf-arica/CuVLER)
9. [2024 arXiv] **Unsupervised Object Localization in the Era of Self-Supervised ViTs: A Survey** [[paper]](https://arxiv.org/pdf/2310.12904)
10. [2024 NIPS] **HASSOD: Hierarchical adaptive self-supervised object detection** [[paper]](https://arxiv.org/pdf/2402.03311) [[code]](https://github.com/Shengcao-Cao/HASSOD)

## Classification based on CLIP
1. [2024 NIPS] **Boosting Vision-Language Models with Transduction** [[paper]](https://arxiv.org/abs/2406.01837) [[code]](https://github.com/MaxZanella/transduction-for-vlms)
2. [2025 CVPR] **Realistic Test-Time Adaptation of Vision-Language Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zanella_Realistic_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2025_paper.pdf) [[code]](https://github.com/MaxZanella/StatA)
3. [2025 CVPR] **COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation** [[paper]](https://arxiv.org/pdf/2503.23388) [[code]](https://github.com/hf618/COSMIC)

## Dataset
1. **VOC**, **ADE20K**, **Context**, **COCO**, **DRIVE** and et al [[download]](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html#drive). Note: You maybe also need helpful [[html]](https://docs.wand-py.org/en/latest/guide/install.html) to install other packages. Conversion codes applied to names in datasets can be found in [[MaskCLIP]](https://github.com/chongzhou96/MaskCLIP/tree/master/tools/convert_datasets).
2. **Imagenet** [[download]](https://image-net.org/challenges/LSVRC/2012/browse-synsets.php). Note: Perhaps you need to **log in** first.
3. **Mini-ImageNet** [[download]](https://pan.baidu.com/s/1MhmzwvzV-hUUMIxKq5z4Gw)(Num 5180). Note: If computing power is not enough to support the Imagenet, you can consider using Mini-ImageNet. Imagenet is about 138G, and Mini-ImageNet is about only 3G.
4. **COD** [[download]](https://pan.baidu.com/s/1ePqUn5vOSZxKG596B-OcNg)(Num 5180). Note: This is the original dataset of Camouflaged Object Detection (COD).
5. **PlantCAMO** [[download]](https://pan.baidu.com/s/1I5igds19w4TqDXppwhDX9Q)(Num 5180). Note: This is COD dataset of plant.
