# Source Code of TESGA

Implementation of Our Paper "Joint multimodal entity-relation extraction based on temporal enhancement and similarity-gated attention" in “Knowledge-Based Systems”.

 [Joint multimodal entity-relation extraction based on temporal enhancement and similarity-gated attention - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705124011389?ref=pdf_download&fr=RR-2&rr=8df208cdac1985d5#page=11&zoom=100,51,356)

## Model Architecture

![image-20241108094255995](https://wgx--img.oss-cn-qingdao.aliyuncs.com/img/image-20241108094255995.png)

(a) The overall structure of the Temporal Enhancement and Similarity-Gated Attention network (TESGA) for multimodal joint entity and relation extraction; (b) Detailed diagram of the dual-channel feature fusion module in the TESGA model, with different colors distinguishing features with different purposes.



## Base on 

> BLIP: [salesforce/BLIP: PyTorch code for BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://github.com/salesforce/BLIP?tab=readme-ov-file)
>
> * `model_base`     `model_base_retrieval_coco`   ` model_base_caption_capfilt_large`
>
> BERT: [google-bert/bert-base-cased · Hugging Face](https://huggingface.co/google-bert/bert-base-cased)
>
> * `bert-base-cased`

## Datesets

> * JMERE :  We use V1 version, please refer to [YuanLi95/EEGA-for-JMERE: This is code for Joint Multimodal Entity-Relation Extraction Based on Edge-enhanced Graph Alignment Network and Word-pair Relation Tagging (AAAI 2023)](https://github.com/YuanLi95/EEGA-for-JMERE?tab=readme-ov-file)
> * MNRE:    [thecharm/Mega: Code for ACM MM 2021 Paper "Multimodal Relation Extraction with Efficient Graph Alignment".](https://github.com/thecharm/Mega?tab=readme-ov-file)
> * Twitter-15

## Pretrained Model

> JMERE-final1.pt
> https://pan.baidu.com/s/186gbvblugbbmurFNqiDRhQ?pwd=5ot2 



## Statement

> Since the author has already graduated and left the research institution, this will be the final update for this project. The methods described in the paper are already quite clear, and the program provided in this project is for reference. I sincerely wish the readers can achieve even more outstanding research results.



## Acknowledgments

[YuanLi95/EEGA-for-JMERE: This is code for Joint Multimodal Entity-Relation Extraction Based on Edge-enhanced Graph Alignment Network and Word-pair Relation Tagging (AAAI 2023)](https://github.com/YuanLi95/EEGA-for-JMERE?tab=readme-ov-file)

[salesforce/BLIP: PyTorch code for BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://github.com/salesforce/BLIP?tab=readme-ov-file)

