# ChatEarthNet: A Global-Scale Image-Text Dataset Empowering Vision-Language Geo-Foundation Models
### Access the dataset
The ChatEarthNet can be downloaded from https://doi.org/10.5281/zenodo.11003436

### Introduction
The Python code utilizes the ChatGPT API to generate captions.

[ChatEarthNet](https://arxiv.org/abs/2402.11325) is a new image-text dataset, providing high-quality natural language descriptions for global-scale satellite data. Specifically, we utilize Sentinel-2 data for its global coverage as the foundational image source, employing semantic segmentation labels from the European Space Agency's WorldCover project to enrich the descriptions of land covers. By conducting in-depth semantic analysis, we formulate detailed prompts to elicit rich descriptions from ChatGPT. We then include a manual verification process to enhance the dataset's quality further. Finally, we offer the community ChatEarthNet, a large-scale image-text dataset characterized by global coverage, high quality, wide-ranging diversity, and detailed descriptions. ChatEarthNet consists of 163,488 image-text pairs with captions generated by ChatGPT-3.5 and an additional 10,000 image-text pairs with captions generated by ChatGPT-4V(ision). This dataset has significant potential for both training and evaluating vision-language geo-foundation models for remote sensing. 

![Example Image](https://github.com/zhu-xlab/ChatEarthNet/blob/main/dataset_vis_1.png)

![Example Image](https://github.com/zhu-xlab/ChatEarthNet/blob/main/dataset_vis_2.png)


# If you find this helpful, please give us a <font color='orange'>STAR ⭐</font>. Thank you, and have a nice day:)

### License
This repository is released under the Apache 2.0 license. The dataset and pretrained model weights are released under the CC-BY-4.0 license.


### Citation
```
@article{yuan2024chatearthnet,
  title={ChatEarthNet: A Global-Scale Image-Text Dataset Empowering Vision-Language Geo-Foundation Models},
  author={Yuan, Zhenghang and Xiong, Zhitong and Mou, Lichao and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2402.11325},
  year={2024}
}
```
