# Content
1. [[Training-Free Open-Vocabulary Semantic Segmentation](#Training-Free_OVSS)]
2. [[Training Open-Vocabulary Semantic Segmentation](#Training_OVSS)]
3. [[Zero-Shot Open-Vocabulary Semantic Segmentation](#ZSOVSS)]
4. [[Few-Shot Open-Vocabulary Semantic Segmentation](#FSOVSS)]
5. [[Supervised Semantic Segmentation](#Supervised_Semantic_Segmentation)]
6. [[Weakly Supervised Semantic Segmentation](#Weakly_Supervised_Semantic_Segmentation)]
7. [[Semi-Supervised Semantic Segmentation](#Semi-Supervised_Semantic_Segmentation)]
8. [[Unsupervised Semantic Segmentation](#Unsupervised_Semantic_Segmentation)]
9. [[Open-Vocabulary Object Detection](#Open-Vocabulary_Object_Detection)]
10. [[Supervised Camouflaged Object Detection](#Supervised_Camouflaged_Object_Detection)]
11. [[Weakly Supervised Object Detection](#Weakly_Supervised_Object_Detection)]
12. [[Unsupervised Object Detection](#Unsupervised_Object_Detection)]
13. [[Classification](#Classification)]
14. [[Diffusion Model](#Diffusion_Model)]
15. [[Foundation models in Segmentation and Classification](#Foundation_models)]
16. [[Dataset](#Dataset)]
-----------------------------------------------------------------------------------------------
# Training-Free Open-Vocabulary Semantic Segmentation
<a name="Training-Free_OVSS"></a>
1. [2024 CVPR] **Clip-diy: Clip dense inference yields open-vocabulary semantic segmentation for-free** [[paper]](https://arxiv.org/pdf/2309.14289)
2. [2024 CVPR] **Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation** [[paper]](https://arxiv.org/pdf/2404.06542) [[code]](https://github.com/aimagelab/freeda)
3. [2024 ECCV] **Diffusion Models for Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2306.09316) [[code]](https://github.com/karazijal/ovdiff)
4. [2024 ECCV] **ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference** [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf) [[code]](https://github.com/mc-lan/ClearCLIP)
5. [2024 ECCV] **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference** [[paper]](https://arxiv.org/pdf/2312.01597) [[code]](https://github.com/wangf3014/SCLIP)
6. [2024 ECCV] **Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP)
7. [2024 ECCV] **Proxyclip: Proxy attention improves clip for open-vocabulary segmentation** [[paper]](https://arxiv.org/pdf/2408.04883) [[code]](https://github.com/mc-lan/ProxyCLIP?tab=readme-ov-file) 
8. [2024 ECCV] **Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2407.08268) [[code]](https://github.com/leaves162/CLIPtrase)
9. [2024 arXiv] **CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13836) [[code]](https://github.com/linsun449/cliper.code?tab=readme-ov-file)
10. [2025 CVPR] **LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.19777) [[code]](https://github.com/vladan-stojnic/LPOSS)
11. [2025 CVPR] **ResCLIP: Residual Attention for Training-free Dense Vision-language Inference** [[paper]](https://arxiv.org/pdf/2411.15851) [[code]](https://github.com/yvhangyang/ResCLIP?tab=readme-ov-file) [[note]](https://www.guyuehome.com/detail?id=1915057149535399937)
12. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Distilling_Spectral_Graph_for_Object-Context_Aware_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/MICV-yonsei/CASS)
13. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person Re-identification** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf) [[code]](https://github.com/Jinxi-Yang-WHU/Fast-GCR.git) [[note]](本文提到的很多re-ranking的技术就是对直接计算的相似度矩阵进行更新，前面公式搞了一大堆，最后就是一个特征传播。)
14. [2025 CVPR] **ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements** [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/PixFoundation/papers/Aydin_ITACLIP_Boosting_Training-Free_Semantic_Segmentation_with_Image_Text_and_Architectural_CVPRW_2025_paper.pdf) [[code]](https://github.com/m-arda-aydn/ITACLIP)
15. [2025 CVPR] **Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sidhu_Search_and_Detect_Training-Free_Long_Tail_Object_Detection_via_Web-Image_CVPR_2025_paper.pdf) [[code]](https://github.com/Mankeerat/SearchDet)
16. [2025 CVPR] **Shift the Lens: Environment-Aware Unsupervised Camouflaged Object Detection** [[paper]]([https://github.com/xiaohainku/EASE](https://openaccess.thecvf.com/content/CVPR2025/papers/Du_Shift_the_Lens_Environment-Aware_Unsupervised_Camouflaged_Object_Detection_CVPR_2025_paper.pdf)) [[code]](https://github.com/xiaohainku/EASE)
17. [2025 ICCV] **LUDVIG: Learning-free Uplifting of 2D Visual features to Gaussian Splatting scene** [[paper]](https://arxiv.org/pdf/2410.14462#page=17.85) [[code]](https://github.com/naver/ludvig)
18. [2025 ICCV] **WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images** [[paper]](https://arxiv.org/pdf/2503.08407)
19. [2025 ICCV] **Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.09219) [[code]](https://github.com/YuHengsss/Trident)
20. [2025 ICCV] **E-SAM: Training-Free Segment Every Entity Model** [[paper]](https://arxiv.org/pdf/2503.12094)
21. [2025 ICCV] **ReME: A Data-Centric Framework for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2506.21233) [[code]](https://github.com/xiweix/ReME)
22. [2025 ICCV] **CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.10086) [[code]](https://github.com/zdk258/CorrCLIP)
23. [2025 ICCV] **CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting** [[paper]](https://arxiv.org/pdf/2505.20469) [[code]](https://epsilontl.github.io/CCL-LGS/)
24. [2025 ICCV] **Auto-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2312.04539)
25. [2025 ICCV] **Understanding Personal Concept in Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2507.11030)
26. [2025 ICCV] **Training-Free Class Purification for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2508.00557)
27. [2025 ICCV] **DIH-CLIP: Unleashing the Diversity of Multi-Head Self-Attention for Training-Free Open-Vocabulary Semantic Segmentation**
28. [2025 ICCV] **Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild**
29. [2025 ICCV] **Feature Purification Matters: Suppressing Outlier Propagation for Training-Free Open-Vocabulary Semantic Segmentation**
30. [2025 ICCV] **Plug-in Feedback Self-adaptive Attention in CLIP for Training-free Open-Vocabulary Segmentation**
31. [2025 ICCV] **Test-Time Retrieval-Augmented Adaptation for Vision-Language Models** [[paper]](https://openreview.net/pdf?id=V3zobHnS61) [[code]](https://github.com/xinqi-fan/TT-RAA)
32. [2025 ICCV] **Text-guided Visual Prompt DINO for Generic Segmentation**
33. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation** [[paper]](https://arxiv.org/pdf/2507.12857) [[code]](https://github.com/HuangShiqi128/SCORE)
34. [2025 ICCV] **Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation**
35. [2025 AAAI] **Training-free Open-Vocabulary Semantic Segmentation via Diverse Prototype Construction and Sub-region Matching** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33137)
36. [2025 arXiv] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.15869) [[code]](https://github.com/SuleBai/SC-CLIP?tab=readme-ov-file)
37. [2025 arXiv] **Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.21844v1) [[code]](https://github.com/dosowiechi/MLMP?tab=readme-ov-file)
38. [2025 arXiv] **FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/abs/2504.10487) [[code]](https://github.com/yasserben/FLOSS)
39. [2025 arXiv] **TextRegion: Text-Aligned Region Tokens from Frozen Image-Text Models** [[paper]](https://arxiv.org/pdf/2505.23769) [[code]](https://github.com/avaxiao/TextRegion)
40. [2025 arXiv] **A Survey on Training-free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.22209)
41. [2025 arXiv] **No time to train! Training-Free Reference-Based Instance Segmentation** [[paper]](https://arxiv.org/pdf/2507.02798) [[code]](https://github.com/miquel-espinosa/no-time-to-train?tab=readme-ov-file) [[note]](https://mp.weixin.qq.com/s/6BjzduZoqc2OgocNkaDiPQ)
# Training Open-Vocabulary Semantic Segmentation
<a name="Training_OVSS"></a>
1. [2022 CVPR] **GroupViT: Semantic Segmentation Emerges from Text Supervision** [[paper]](https://arxiv.org/pdf/2202.11094) [[code]](https://github.com/NVlabs/GroupViT?tab=readme-ov-file)
2. [2023 CVPR] **Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning** [[paper]](https://arxiv.org/pdf/2212.04994)
3. [2023 CVPR] **Learning to Generate Text-grounded Mask for Open-world Semantic Segmentation from Only Image-Text Pairs** [[paper]](https://arxiv.org/pdf/2212.00785) [[code]](https://github.com/khanrc/tcl?tab=readme-ov-file)
4. [2023 ICCV] **Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.pdf)
5. [2023 ICML] **SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2211.14813) [[code]](https://github.com/ArrowLuo/SegCLIP?tab=readme-ov-file)
6. [2023 ICML] **Grounding Everything: Emerging Localization Properties in Vision-Language Transformers** [[paper]](https://arxiv.org/pdf/2312.00878) [[code]](https://github.com/WalBouss/GEM?tab=readme-ov-file)
8. [2024 CVPR] **SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2311.15537v2) [[code]](https://github.com/xb534/SED.git)
9. [2024 CVPR] **Not All Classes Stand on Same Embeddings: Calibrating a Semantic Distance with Metric Tensor** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10658112&tag=1)
10. [2024 CVPR] **USE: Universal Segment Embeddings for Open-Vocabulary Image Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_USE_Universal_Segment_Embeddings_for_Open-Vocabulary_Image_Segmentation_CVPR_2024_paper.pdf)
11. [2024 CVPR] **CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2303.11797) [[code]](https://github.com/cvlab-kaist/CAT-Seg)
12. [2024 CVPR] **Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Emergent_Open-Vocabulary_Semantic_Segmentation_from_Off-the-shelf_Vision-Language_Models_CVPR_2024_paper.pdf) [[code]](https://github.com/letitiabanana/PnP-OVSS)
13. [2024 CVPR] **SAM-CLIP: Merging Vision Foundation Models towards Semantic and Spatial Understanding** [[paper]](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Wang_SAM-CLIP_Merging_Vision_Foundation_Models_Towards_Semantic_and_Spatial_Understanding_CVPRW_2024_paper.pdf)
14. [2024 CVPR] **Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Image-to-Image_Matching_via_Foundation_Models_A_New_Perspective_for_Open-Vocabulary_CVPR_2024_paper.pdf)
16. [2024 ECCV] **CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation** [[paper]](https://arxiv.org/pdf/2312.12359) [[code]](https://github.com/wysoczanska/clip_dinoiser)
17. [2024 ECCV] **In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation** [[paper]](http://arxiv.org/abs/2408.04961) [[code]](https://github.com/dahyun-kang/lavg)
18. [2024 ICLR] **CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction** [[paper]](https://arxiv.org/pdf/2310.01403v2) [[code]](https://github.com/wusize/CLIPSelf?tab=readme-ov-file)
19. [2024 NIPS] **Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1119587863e78451f080da2a768c4935-Paper-Conference.pdf) [[code]](https://github.com/cvlab-kaist/PixelCLIP)
20. [2024 NIPS] **Relationship Prompt Learning is Enough for Open-Vocabulary Semantic Segmentation** [[paper]](https://openreview.net/pdf?id=PKcCHncbzg)
21. [2024 arXiv] **DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment** [[paper]](https://arxiv.org/pdf/2412.16334)
22. [2025 CVPR] **Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.21780)
23. [2025 CVPR] **Your ViT is Secretly an Image Segmentation Model** [[paper]](https://arxiv.org/pdf/2503.19108) [[code]](https://github.com/tue-mps/eomt)
24. [2025 CVPR] **Exploring Simple Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lai_Exploring_Simple_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf)
25. [2025 CVPR] **Dual Semantic Guidance for Open Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Dual_Semantic_Guidance_for_Open_Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf)
27. [2025 CVPR] **DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DeCLIP_Decoupled_Learning_for_Open-Vocabulary_Dense_Perception_CVPR_2025_paper.pdf) [[code]](https://github.com/xiaomoguhz/DeCLIP)
28. [2025 ICCV] **Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.19331) [[code]](https://github.com/lorebianchi98/Talk2DINO)
29. [2025 ICLR] **Cross the Gap: Exposing the Intra-modal Misalignment in CLIP via Modality Inversion** [[paper]](https://arxiv.org/pdf/2502.04263) [[code]](https://github.com/miccunifi/Cross-the-Gap?tab=readme-ov-file)
# Zero-Shot Open-Vocabulary Semantic Segmentation
<a name="ZSOVSS"></a>
1. [2024 CVPR] **On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_On_the_Test-Time_Zero-Shot_Generalization_of_Vision-Language_Models_Do_We_CVPR_2024_paper.pdf) [[code]](https://github.com/MaxZanella/MTA)
2. [2024 CVPR] **Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10656627&tag=1) [[code]](https://github.com/Jittor/JSeg)
3. [2024 CVPR] **Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Diffuse_Attend_and_Segment_Unsupervised_Zero-Shot_Segmentation_using_Stable_Diffusion_CVPR_2024_paper.pdf) [[code]](https://github.com/google/diffseg)
4. [2024 ECCV] **OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.14183) [[code]](https://github.com/cubeyoung/OTSeg?tab=readme-ov-file)
5. [2024 ICCV] **Zero-guidance Segmentation Using Zero Segment Labels** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Rewatbowornwong_Zero-guidance_Segmentation_Using_Zero_Segment_Labels_ICCV_2023_paper.pdf) [[code]](https://github.com/nessessence/ZeroGuidanceSeg)
6. [2024 NIPS] **DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1867748a011e1425b924ec72a4066b62-Paper-Conference.pdf) [[code]](https://github.com/PaulCouairon/DiffCut)
7. [2025 ICLR] **Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model** [[paper]](https://arxiv.org/pdf/2412.18303) [[code]](https://github.com/Yushu-Li/ECALP?tab=readme-ov-file)
# Few-Shot Open-Vocabulary Semantic Segmentation
<a name="FSOVSS"></a>
1. [2024 NIPS] **Training-Free Open-Ended Object Detection and Segmentation via Attention as Prompts** [[paper]](https://arxiv.org/pdf/2410.05963)
2. [2024 NIPS] **A Surprisingly Simple Approach to Generalized Few-Shot Semantic Segmentation** [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f75a57e9c71e8369da0150ea769d5a2-Paper-Conference.pdf) [[code]](https://github.com/IBM/BCM)
3. [2024 NIPS] **Renovating Names in Open-Vocabulary Segmentation Benchmarks** [[paper]](https://openreview.net/pdf?id=Uw2eJOI822) [[code]](https://andrehuang.github.io/renovate/)
4. [2025 CVPR] **Hyperbolic Uncertainty-Aware Few-Shot Incremental Point Cloud Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sur_Hyperbolic_Uncertainty-Aware_Few-Shot_Incremental_Point_Cloud_Segmentation_CVPR_2025_paper.pdf)
5. [2025 ICCV] **Probabilistic Prototype Calibration of Vision-language Models for Generalized Few-shot Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2506.22979) [[code]](https://github.com/jliu4ai/FewCLIP)
7. [2025 MICCAI] **Realistic Adaptation of Medical Vision-Language Models** [[paper]](https://arxiv.org/pdf/2506.17500) [[code]](https://github.com/jusiro/SS-Text)

# Supervised Semantic Segmentation
<a name="Supervised_Semantic_Segmentation"></a>
1. [2021 ICCV] **Vision Transformers for Dense Prediction** [[paper]](https://arxiv.org/abs/2103.13413v1) [[code]](https://github.com/isl-org/DPT?tab=readme-ov-file)
2. [2021 ICCV] **Segmenter: Transformer for Semantic Segmentation** [[paper]](https://arxiv.org/abs/2105.05633) [[code]](https://github.com/rstrudel/segmenter)
3. [2022 ICLR] **Language-driven Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2201.03546) [[code]](https://github.com/isl-org/lang-seg)
4. [2025 CVPV] **Your ViT is Secretly an Image Segmentation Model** [[paper]](https://arxiv.org/pdf/2503.19108) [[code]](https://github.com/tue-mps/eomt)
# Weakly Supervised Semantic Segmentation
<a name="Weakly_Supervised_Semantic_Segmentation"></a>
1. [2021 NIPS] **Looking Beyond Single Images for Contrastive Semantic Segmentation Learning** [[paper]](https://proceedings.neurips.cc/paper/2021/file/1a68e5f4ade56ed1d4bf273e55510750-Paper.pdf)
2. [2022 CVPR] **Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers** [[paper]](https://arXiv.org/pdf/2203.02664) [[code]](https://github.com/rulixiang/afa)
3. [2022 CVPR] **MCTFormer:Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2203.02891) [[code]](https://github.com/xulianuwa/MCTformer)
4. [2022 IEEE] **Looking Beyond Single Images for Weakly Supervised Semantic Segmentation Learning**  [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9760057) [[code]](https://github.com/GuoleiSun/MCIS_wsss)
5. [2023 CVPR] **Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf) [[code]](https://github.com/xulianuwa/MMCST)
6. [2023 ICCV] **Spatial-Aware Token for Weakly Supervised Object Localization** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf) [[code]](https://github.com/wpy1999/SAT)
7. [2023 CVPR] **Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentatio** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Rong_Boundary-Enhanced_Co-Training_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf) [[code]](https://github.com/ShenghaiRong/BECO?tab=readme-ov-file)
8. [2023 CVPR] **ToCo:Token Contrast for Weakly-Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2303.01267) [[code]](https://github.com/rulixiang/ToCo)
9. [2023 CVPR] **CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2212.09506) [[code]](https://github.com/linyq2117/CLIP-ES)
10. [2023 NIPS] **Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2310.19001) [[code]](https://github.com/Ferenas/PGSeg?tab=readme-ov-file)
11. [2023 arXiv] **MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2308.03005) [[code]](https://github.com/xulianuwa/MCTformer)
12. [2024 CVPR] **Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2406.11189v1) [[code]](https://github.com/zbf1991/WeCLIP)
13. [2024 CVPR] **CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.13147v1) [[code]](https://github.com/ZiqinZhou66/ZegCLIP?tab=readme-ov-file)
14. [2024 CVPR] **DuPL: Dual Student with Trustworthy Progressive Learning for Robust Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.11184) [[code]](https://github.com/Wu0409/DuPL)
15. [2024 CVPR] **Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation** [[paper]](https:https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_Hunting_Attributes_Context_Prototype-Aware_Learning_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/Barrett-python/CPAL)
16. [2024 CVPR] **Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_the_Generalization_of_Segmentation_Foundation_Model_under_Distribution_Shift_CVPR_2024_paper.pdf)
17. [2024 CVPR] **Official code for Class Tokens Infusion for Weakly Supervised Semantic Segmentation** [[paper]](Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper) [[code]](https://github.com/yoon307/CTI)
18. [2024 CVPR] **Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2402.18467) [[code]](https://github.com/zwyang6/SeCo)
19. [2024 CVPR] **Class Tokens Infusion for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) [[code]](https://github.com/yoon307/CTI)
20. [2024 CVPR] **SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2401.11719) [[code]](https://github.com/Barrett-python/SFC)
21. [2024 CVPR] **PSDPM:Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_PSDPM_Prototype-based_Secondary_Discriminative_Pixels_Mining_for_Weakly_Supervised_Semantic_CVPR_2024_paper.pdf) [[code]](https://github.com/xinqiaozhao/PSDPM)
22. [2024 ECCV] **DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2409.15801)
23. [2024 ECCV] **CoSa:Weakly Supervised Co-training with Swapping Assignments for Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2402.17891) [[code]](https://github.com/youshyee/CoSA)
24. [2024 AAAI] **Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2312.08916) [[code]](https://github.com/Jessie459/feature-self-reinforcement)
25. [2024 arXiv] **A Realistic Protocol for Evaluation of Weakly Supervised Object Localization** [[paper]](https://arXiv.org/pdf/2404.10034) [[code]](https://github.com/shakeebmurtaza/wsol_model_selection)
26. [2024 IEEE] **SSC:Spatial Structure Constraints for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2401.11122) [[code]](https://github.com/NUST-Machine-Intelligence-Laboratory/SSC)
27. [2024 IEEE] **Modeling the label distributions for weakly-supervised semantic segmentation** [[paper]](https://arxiv.org/pdf/2403.13225) [[code]](https://github.com/Luffy03/AGMM-SASS)
28. [2025 CVPR] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/jianwang91/POT)
29. [2025 CVPR] **PROMPT-CAM: A Simpler Interpretable Transformer for Fine-Grained Analysis** [[paper]](https://arXiv.org/pdf/2501.09333) [[code]](https://github.com/Imageomics/Prompt_CAM)
30. [2025 CVPR] **Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2503.20826) [[code]](https://github.com/zwyang6/ExCEL)
31. [2025 CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
32. [2025 CVPR] **Prompt Categories Cluster for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2412.13823)
33. [2025 ICCV] **Class Token as Proxy: Optimal Transport-assisted Proxy Learning for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wang_Class_Token_as_ICCV_2025_supplemental.pdf) 
34. [2025 ICCV] **Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2507.06848) [[code]](https://github.com/HSG-AIML/TokenMasking-WSSS)
35. [2025 ICCV] **Bias-Resilient Weakly Supervised Semantic Segmentation Using Normalizing Flows** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.pdf) [[code]](https://github.com/DpDark/BRNF)
36. [2025 ICCV] **OVA-Fields: Weakly Supervised Open-Vocabulary Affordance Fields for Robot Operational Part Detection** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Su_OVA-Fields_Weakly_Supervised_Open-Vocabulary_Affordance_Fields_for_Robot_Operational_Part_ICCV_2025_paper.pdf) [[code]](https://github.com/vlasu19/OVA-Fields)
37. [2025 AAAI] **MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2412.11076) [[code]](https://github.com/zwyang6/MoRe)
38. [2025 arXiv] **TeD-Loc: Text Distillation for Weakly Supervised Object Localization** [[paper]](https://arXiv.org/pdf/2501.12632) [[code]](https://github.com/shakeebmurtaza/TeDLOC)
39. [2025 arXiv] **Image Augmentation Agent for Weakly Supervised Semantic Segmentation** [[paper]](https://arXiv.org/pdf/2412.20439)
# Semi-Supervised Semantic Segmentation
<a name="Semi-Supervised_Semantic_Segmentation"></a>
1. [2025 ICCV] **ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction** [[paper]](https://arxiv.org/pdf/2507.15803) [[code]](https://github.com/xinqi-fan/TT-RAA)
# Unsupervised Semantic Segmentation
<a name="Unsupervised_Semantic_Segmentation"></a>
1. [2021 ICCV] **Emerging Properties in Self-Supervised Vision Transformers** [[paper]](http://openaccess.thecvf.com//content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [[code]](https://github.com/facebookresearch/dino) [[note]](https://blog.csdn.net/YoooooL_/article/details/129234966)
2. [2022 CVPR] **Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization** [[paper]](https://arxiv.org/abs/2205.07839) [[code]](https://github.com/lukemelas/deep-spectral-segmentation?tab=readme-ov-file)
3. [2022 CVPR] **Freesolo: Learning to segment objects without annotations** [[paper]](https://arxiv.org/pdf/2202.12181) [[code]](https://github.com/NVlabs/FreeSOLO)
4. [2022 ECCV] **Extract Free Dense Labels from CLIP** [[paper]](https://arxiv.org/pdf/2112.01071) [[code]](https://github.com/chongzhou96/MaskCLIP?tab=readme-ov-file) [[note]](https://www.cnblogs.com/lipoicyclic/p/16967704.html)
5. [2023 CVPR] **ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation** [[paper]](https://arxiv.org/abs/2212.03588) [[code]](https://github.com/ZiqinZhou66/ZegCLIP?tab=readme-ov-file)
6. [2024 CVPR] **Guided Slot Attention for Unsupervised Video Object Segmentation** [[paper]](https://arxiv.org/pdf/2303.08314v3) [[code]](https://github.com/Hydragon516/GSANet)
7. [2024 CVPR] **ReCLIP++:Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2408.06747) [[code]](https://github.com/dogehhh/ReCLIP)
8. [2024 CVPR] **CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers** [[paper]](https://arxiv.org/pdf/2403.07700) [[code]](https://github.com/shahaf-arica/CuVLER?tab=readme-ov-file)
9. [2024 CVPR] **EAGLE: Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2403.01482) [[code]](https://github.com/MICV-yonsei/EAGLE?tab=readme-ov-file)
10. [2024 ECCV] **Unsupervised Dense Prediction using Differentiable Normalized Cuts** [[paper]](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05675.pdf)
11. [2024 NIPS] **PaintSeg: Training-free Segmentation via Painting** [[paper]](https://arxiv.org/abs/2305.19406)
12. [2025 ICCV] **DIP: Unsupervised Dense In-Context Post-training of Visual Representations** [[paper]](https://arxiv.org/pdf/2506.18463) [[code]](https://github.com/sirkosophia/DIP)

# Open-Vocabulary Object Detection
<a name="Open-Vocabulary_Object_Detection"></a>
1. [2022 ICLR] **Open-vocabulary Object Detection via Vision and Language Knowledge Distillation** [[paper]](https://arxiv.org/pdf/2104.13921) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
2. [2022 CVPR] **Grounded Language-Image Pre-training** [[paper]](https://arxiv.org/pdf/2112.03857) [[code]](https://github.com/microsoft/GLIP)
3. [2024 CVPR] **SHiNe: Semantic Hierarchy Nexus for Open-vocabulary Object Detection** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_SHiNe_Semantic_Hierarchy_Nexus_for_Open-vocabulary_Object_Detection_CVPR_2024_paper.pdf)
4. [2024 ECCV] **Grounding dino: Marrying dino with grounded pre-training for open-set object detection** [[paper]](https://arxiv.org/pdf/2303.05499) [[code]](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file) [[note]](https://blog.csdn.net/xhtchina/article/details/147641112)
5. [2025 ICCV] **Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes** [[paper]](https://www.arxiv.org/pdf/2508.02157) [[code]](https://github.com/Fischer-Tom/unified-detection-and-pose-estimation)
# Supervised Camouflaged Object Detection
<a name="Supervised_Camouflaged_Object_Detection"></a>
1. [2020 CVPR] **Camouflaged Object Detection** [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.pdf) [[code]](https://github.com/DengPingFan/SINet)
2. [2023 AAAI] **High-resolution Iterative Feedback Network for Camouflaged Object Detection** [[paper]](https://arxiv.org/pdf/2203.11624) [[code]](https://github.com/HUuxiaobin/HitNet?tab=readme-ov-file)
3. [2024 arXiv] **PlantCamo: Plant Camouflage Detection** [[paper]](https://arxiv.org/pdf/2410.17598v1) [[code]](https://github.com/yjybuaa/PlantCamo)
# Weakly Supervised Object Detection
<a name="Weakly_Supervised_Object_Detection"></a>
1. [2023 CVPR] **Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Multi-Modal_Class-Specific_Tokens_for_Weakly_Supervised_Dense_Object_Localization_CVPR_2023_paper.pdf) [[code]](https://github.com/xulianuwa/MMCST)
2. [2023 ICCV] **Spatial-Aware Token for Weakly Supervised Object Localization** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Spatial-Aware_Token_for_Weakly_Supervised_Object_Localization_ICCV_2023_paper.pdf) [[code]](https://github.com/wpy1999/SAT?tab=readme-ov-file)
# Unsupervised Object Detection
<a name="Unsupervised_Object_Detection"></a>
1. [2022 CVPR] **Self-supervised transformers for unsupervised object discovery using normalized cut** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.pdf) [[code]](https://www.m-psi.fr/Papers/TokenCut2022/)
2. [2023 CVPR] **Unsupervised Object Localization: Observing the Background to Discover Objects** [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Simeoni_Unsupervised_Object_Localization_Observing_the_Background_To_Discover_Objects_CVPR_2023_paper.pdf) [[code]](https://github.com/valeoai/FOUND)
3. [2023 CVPR] **Weakly-supervised Contrastive Learning for Unsupervised Object Discovery** [[paper]](https://arxiv.org/pdf/2307.03376) [[code]](https://github.com/npucvr/WSCUOD)
4. [2024 CVPR] **DIOD: Self-Distillation Meets Object Discovery** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Kara_DIOD_Self-Distillation_Meets_Object_Discovery_CVPR_2024_paper.pdf) [[code]](https://github.com/CEA-LIST/DIOD)
5. [2023 ICCV] **MOST: multiple object local- ization with self-supervised transformers for object discovery** [[paper]]()
6. [2023 ICCV] **SEMPART: Self-supervised Multi-resolution Partitioning of Image Semantics** [[paper]](https://arxiv.org/pdf/2309.10972)
7. [2023 ICCV] **Box-based Refinement for Weakly Supervised and Unsupervised Localization Tasks** [[paper]](https://arxiv.org/pdf/2309.03874) [[code]](https://github.com/eyalgomel/box-based-refinement)
8. [2024 CVPR] **CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers** [[paper]](https://arxiv.org/pdf/2403.07700) [[code]](https://github.com/shahaf-arica/CuVLER)
9. [2024 NIPS] **HASSOD: Hierarchical adaptive self-supervised object detection** [[paper]](https://arxiv.org/pdf/2402.03311) [[code]](https://github.com/Shengcao-Cao/HASSOD)
10. [2024 arXiv] **Unsupervised Object Localization in the Era of Self-Supervised ViTs: A Survey** [[paper]](https://arxiv.org/pdf/2310.12904)
11. [2025 ICCV] **Ensemble Foreground Management for Unsupervised Object Discovery** [[paper]](https://arxiv.org/pdf/2507.20860) [[code]](https://github.com/YFaris/UnionCut)
12. [2025 arXiv] **RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models** [[paper]](https://arxiv.org/pdf/2510.25257v1) [[code]](https://github.com/RT-DETRs/RT-DETRv4) [[note]](https://zhuanlan.zhihu.com/p/1967226872959050195)

# Classification
<a name="Classification"></a>
1. [2023 ICCV] **Black Box Few-Shot Adaptation for Vision-Language models** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.pdf) [[code]](https://github.com/saic-fi/LFA)
2. [2024 NIPS] **Boosting Vision-Language Models with Transduction** [[paper]](https://arxiv.org/abs/2406.01837) [[code]](https://github.com/MaxZanella/transduction-for-vlms)
3. [2024 NIPS] **Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting** [[paper]](https://arxiv.org/pdf/2410.19294v1) [[code]](https://github.com/zhuhsingyuu/Frolic)
4. [2024 ICLR] **A Hard-to-Beat Baseline for Training-free CLIP-Based Adaptation** [[paper]](https://arxiv.org/pdf/2402.04087) [[code]](https://github.com/mrflogs/ICLR24)
5. [2025 CVPR] **Realistic Test-Time Adaptation of Vision-Language Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zanella_Realistic_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2025_paper.pdf) [[code]](https://github.com/MaxZanella/StatA)
6. [2025 CVPR] **COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation** [[paper]](https://arxiv.org/pdf/2503.23388) [[code]](https://github.com/hf618/COSMIC)
7. [2025 CVPR] **Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Few-Shot_Recognition_via_Stage-Wise_Retrieval-Augmented_Finetuning_CVPR_2025_paper.pdf) [[code]](https://tian1327.github.io/SWAT)
8. [2025 CVPR] **ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models** [[paper]](https://arxiv.org/pdf/2501.11175) [[code]](https://github.com/ybendou/ProKeR)
9. [2025 CVPR] **GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery** [[paper]](https://arXiv.org/abs/2403.09974) [[code]](https://github.com/enguangW/GET)
10. [2025 ICCV] **Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation** [[paper]](https://arxiv.org/pdf/2410.14729)
11. [2025 ICCV] **Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models** [[paper]](https://arxiv.org/pdf/2508.01225) [[code]](https://github.com/CenturyChen/MCP)
12. [2025 ICLR] **RA-TTA: Retrieval-Augmented Test-Time Adaptation for Vision-Language Models** [[paper]](https://openreview.net/pdf?id=V3zobHnS61) [[code]](https://github.com/kaist-dmlab/RA-TTA)
13. [2025 TMLR] **Memory-Modular Classification: Learning to Generalize with Memory Replacement** [[paper]](https://openreview.net/pdf?id=DcIW0idrg8) [[code]](https://github.com/dahyun-kang/mml)
14. [2025 arXiv] **Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment** [[paper]](https://arxiv.org/pdf/2508.15568) [[code]](https://github.com/AIM-SKKU/ADAPT)

# Diffusion Model
<a name="Diffusion_Model"></a>
1. [2022 CVPR] **High-Resolution Image Synthesis with Latent Diffusion Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) [[code]](https://github.com/CompVis/latent-diffusion)
2. [2023 ICLR] **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** [[paper]](https://arxiv.org/pdf/2307.01952) [[code]](https://github.com/Stability-AI/generative-models)
3. [2023 CVPR] **Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning** [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Feng_Diverse_Data_Augmentation_with_Diffusions_for_Effective_Test-time_Prompt_Tuning_ICCV_2023_paper.pdf) [[code]](https://github.com/chunmeifeng/DiffTPT?tab=readme-ov-file)
4. [2025 ICCV] **Dataset Distillation via Vision-Language Category Prototype** [[paper]](https://arxiv.org/pdf/2506.23580) [[code]](https://github.com/zou-yawen/Dataset-Distillation-via-Vision-Language-Category-Prototype/)

# Foundation models in Segmentation and Classification
<a name="Foundation_models"></a>
1. [2021 ICML] **CLIP: Learning Transferable Visual Models From Natural Language Supervision** [[paper]](https://arxiv.org/pdf/2103.00020) [[code]](https://github.com/openai/CLIP)
2. [2021 ICCV] **DINO: Emerging Properties in Self-Supervised Vision Transformers** [[paper]](https://arxiv.org/pdf/2104.14294) [[code]](https://github.com/facebookresearch/dino)
3. [2024 TMLR] **DINOv2: Learning robust visual features without supervision** [[paper]](https://arxiv.org/pdf/2304.07193) [[code]](https://github.com/facebookresearch/dinov2)
4. [2025 arXiv] **DINOv3** [[paper]](https://arxiv.org/pdf/2508.10104) [[code]](https://github.com/facebookresearch/dinov3)
5. [2025 arXiv] **SAM3: SegmentAnythingwithConcepts** [[paper]](https://arxiv.org/pdf/2511.16719) [[code]](https://github.com/facebookresearch/sam3)

# Dataset
1. **VOC**, **ADE20K**, **Context**, **COCO**, **DRIVE** and et al [[download]](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html#drive). Note: You maybe also need helpful [[html]](https://docs.wand-py.org/en/latest/guide/install.html) to install other packages. Conversion codes applied to names in datasets can be found in [[MaskCLIP]](https://github.com/chongzhou96/MaskCLIP/tree/master/tools/convert_datasets).
2. **Imagenet** [[download]](https://image-net.org/challenges/LSVRC/2012/browse-synsets.php). Note: Perhaps you need to **log in** first.
3. **Mini-ImageNet** [[download]](https://pan.baidu.com/s/1MhmzwvzV-hUUMIxKq5z4Gw)(Num 5180). Note: If computing power is not enough to support the Imagenet, you can consider using Mini-ImageNet. Imagenet is about 138G, and Mini-ImageNet is about only 3G.
4. **COD** [[download]](https://pan.baidu.com/s/1ePqUn5vOSZxKG596B-OcNg)(Num 5180). Note: This is the original dataset of Camouflaged Object Detection (COD).
5. **PlantCAMO** [[download]](https://pan.baidu.com/s/1I5igds19w4TqDXppwhDX9Q)(Num 5180). Note: This is COD dataset of plant.
