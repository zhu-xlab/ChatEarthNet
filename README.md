# ChatEarthNet
We introduce a new image-text dataset, providing high-quality natural language descriptions for global-scale satellite data. Specifically, we utilize Sentinel-2 data for its global coverage as the foundational image source, employing semantic segmentation labels from the European Space Agency's (ESA) WorldCover project to enrich the descriptions of land covers. By conducting in-depth semantic analysis, we formulate detailed prompts to elicit rich descriptions from ChatGPT. We then include a manual verification process to enhance the dataset's quality further. This step involves manual inspection and correction to refine the dataset. Finally, we offer the community ChatEarthNet, a large-scale image-text dataset characterized by global coverage, high quality, wide-ranging diversity, and detailed descriptions. ChatEarthNet consists of 163,488 image-text pairs with captions generated by ChatGPT-3.5 and an additional 10,000 image-text pairs with captions generated by ChatGPT-4V(ision). This dataset has significant potential for both training and evaluating vision-language geo-foundation models for remote sensing. 

![Example Image](images/dataset_vis.pdf "An overview of the ChatEarthNet dataset. We randomly select image-text samples from four different locations. The left and top sides display the descriptions generated by ChatGPT-4V. While the right and bottom sides show two samples produced by ChatGPT-3.5. We use different colors to highlight the words of different land cover types.")
