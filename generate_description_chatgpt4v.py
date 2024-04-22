# analyze the image and generate the corresponding prompts

import numpy as np
from PIL import Image
import base64
import requests
import io
import json
import time



land_cover_label = {'land_cover': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'background',
            'water', 'developed', 'tree', 'shrub', 'grass',
            'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss',
        ],
        'colors': [
            [0, 0, 0], # unknown
            [0, 0, 255], # (blue) water
            [255, 0, 0], # (red) developed
            [0, 192, 0], # (dark green) tree
            [200, 170, 120], # (brown) shrub
            [0, 255, 0], # (green) grass
            [255, 255, 0], # (yellow) crop
            [128, 128, 128], # (grey) bare
            [255, 255, 255], # (white) snow
            [0, 255, 255], # (cyan) wetland
            [255, 0, 255], # (pink) mangroves
            [128, 0, 128], # (purple) moss
        ],
 }}

labels = land_cover_label['land_cover']['categories']
colors = land_cover_label['land_cover']['colors']



def analyze_segmentation_map(image):
    h, w = image.shape
    patches = {
        "top_left": image[:h//2, :w//2],
        "top_right": image[:h//2, w//2:],
        "bottom_left": image[h//2:, :w//2],
        "bottom_right": image[h//2:, w//2:],
        "middle": image[h//4:3*h//4, w//4:3*w//4]
    }

    analyze_prompt = ""

    for name, patch in patches.items():
        unique, counts = np.unique(patch, return_counts=True)
        proportions = counts / counts.sum()
        sorted_indices = np.argsort(-proportions)  # Sorting in descending order

        statistic_str = f"{name} distribution:"
        for idx in sorted_indices:
            if unique[idx] == 0:
                continue
            statistic_str += f" {labels[unique[idx]]}: {proportions[idx]:.2f};"
        analyze_prompt += statistic_str
        analyze_prompt += '\n'
    return analyze_prompt



import numpy as np
from PIL import Image
from collections import Counter


def split_into_patches(image):
    h, w = image.shape[:2]
    patches = {
        "top left part": image[:h//2, :w//2],
        "top right part": image[:h//2, w//2:],
        "bottom left part": image[h//2:, :w//2],
        "bottom right part": image[h//2:, w//2:],
        "middle part": image[h//4:h*3//4, w//4:w*3//4]
    }
    return patches

def count_pixel_proportions(num, land_type, patch):
    count = np.count_nonzero(patch == land_type)
    proportions = count / (128.*128.)
    return proportions

def analyze_image(image):
    patches = split_into_patches(image)
    all_land_types = Counter(list(image.reshape(-1)))
    analyze_prompt = ""

    for land_type, num in all_land_types.items():
        if land_type == 0:
            continue
        analyze_prompt += f"{labels[land_type]}: "
        for patch_name, patch in patches.items():
            proportion = count_pixel_proportions(num, land_type, patch)
            analyze_prompt += f" {patch_name}: {proportion:.2%} "
        analyze_prompt += '\n'
    return analyze_prompt


import numpy as np

def convert_color_map_to_segmentation(color_map, label_colors):
    """
    Convert a color map back to a segmentation map with pixel labels from 0 to 7.

    Args:
    color_map (numpy.ndarray): A 3D array where each element represents the color of a pixel.
    label_colors (list): A list of color tuples corresponding to each label.

    Returns:
    numpy.ndarray: A 2D array representing the segmentation map.
    """
    # Initialize an empty segmentation map with the same height and width as the color map
    segmentation_map = np.zeros((color_map.shape[0], color_map.shape[1]), dtype=np.uint8)

    # Map each color in the color map back to its corresponding label
    for label, color in enumerate(label_colors):
        # Create a mask where the color matches the current label color
        mask = np.all(color_map == color, axis=-1)

        # Assign the label to the matching locations in the segmentation map
        segmentation_map[mask] = label

    return segmentation_map

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def downsample_image(image, skip_index):
    """
    Downsample a PIL image by skipping pixels.

    Args:
    image (PIL.Image.Image): The source image.
    skip_index (int): The number of pixels to skip.

    Returns:
    PIL.Image.Image: The downsampled image.
    """
    # Ensure the input is a PIL Image
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image.Image object")

    # Get the size of the original image
    width, height = image.size

    # Calculate the size of the downsampled image
    new_width = (width + skip_index - 1) // skip_index
    new_height = (height + skip_index - 1) // skip_index

    # Create a new image of the desired size
    downsampled_image = Image.new("RGB", (new_width, new_height))

    # Copy pixels from the original image to the new image, skipping as appropriate
    for y in range(0, height, skip_index):
        for x in range(0, width, skip_index):
            downsampled_image.putpixel((x // skip_index, y // skip_index), image.getpixel((x, y)))

    return downsampled_image


def resize_and_encode_image(image_path):
    """
    Resize an image to 128x128 using nearest neighbor interpolation and then encode it to base64.

    Args:
    image_path (str): The path to the image file.

    Returns:
    str: A base64 encoded string of the resized image.
    """
    # Open the image
    with Image.open(image_path) as img:
        # Resize the image
        #resized_img = img.resize((128, 128), Image.NEAREST)
        resized_img = downsample_image(img, 2)

        # Save the resized image to a bytes buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format=img.format)

        # Get the byte data from the buffer
        byte_data = buffer.getvalue()

    # Encode the byte data to base64
    base64_encoded = base64.b64encode(byte_data).decode('utf-8')

    return base64_encoded


def generate_captions(image_paths, segmentation_maps):
    analyze_prompt1 = analyze_segmentation_map(segmentation_maps[0])
    analyze_prompt2 = analyze_segmentation_map(segmentation_maps[1])
    analyze_prompt3 = analyze_segmentation_map(segmentation_maps[2])
    analyze_prompt4 = analyze_segmentation_map(segmentation_maps[3])

    # Example usage
    analyze_prompt21 = analyze_image(segmentation_maps[0])
    analyze_prompt22 = analyze_image(segmentation_maps[1])
    analyze_prompt23 = analyze_image(segmentation_maps[2])
    analyze_prompt24 = analyze_image(segmentation_maps[3])

    prompt = "You are an AI visual assistant that can analyze the given image. In the image, different colors represent different land cover types.\
            The color for the land cover dictionary is: '[0, 0, 255] (blue): water; [255, 0, 0](red): developed area; \
            [0, 192, 0] (dark green): tree; [200, 170, 120] (brown): shrub; [0, 255, 0] (green): grass; [255, 255, 0] (yellow): crop;\
            [128, 128, 128] (grey): bare; [255, 255, 255] (white): snow; [0, 255, 255] (cyan): wetland; [255, 0, 255] (pink): mangroves; [128, 0, 128] (purple): moss.' You will be provided four independent images at once." 

    prompt += "For the first image, the distribution of each land cover type is:"
    prompt += analyze_prompt21
    prompt += "For the first image, the spatial distribution of the image is:"
    prompt += analyze_prompt1
    prompt += "For the second image, the distribution of each land cover type is:"
    prompt += analyze_prompt22
    prompt += "For the second image, the spatial distribution of the image is:"
    prompt += analyze_prompt2
    prompt += "For the third image, the distribution of each land cover type is:"
    prompt += analyze_prompt23
    prompt += "For the third image, the spatial distribution of the image is:"
    prompt += analyze_prompt3
    prompt += "For the fourth image, the distribution of each land cover type is:"
    prompt += analyze_prompt24
    prompt += "For the fourth image, the spatial distribution of the image is:"
    prompt += analyze_prompt4

    prompt += "You are given four independent images, describe in long sentences for each image seperately using four paragraphs and avoid saying other things.\
            The following constraints should be obeyed: \
            1) Do not use color-related words; treat the color as the land cover type directly.\
            2) Generate the four descriptions seperately; do not add connection between them. \
            3) When describing water, developed, and crop areas, incorporate shape descriptors.\
            4) Double-check all the presented land cover types based on the distribution of each land cover type. If some land covers are not presented, do not mention them.\
            5) Describe it objectively; do not use words: 'possibly','likely','perhaps','color dictionary','appear','change','transition', 'dynamic', or any words with similar connotations.\
            6) Double-check the shape and location of the developed area, water course, grass, tree, shrub, wetland, and crop areas based on the given image if they are present.\
            7) Consider the spatial statistics as a unified image without breaking them down into individual spatial distributions and land cover proportions when describing the overall scene.\
            8) Describe each land cover separately for each given image, and then describe the main theme of each given image."

    api_key = "openai api key"

    #base64_image = encode_image(image_path)
    base64_image1 = resize_and_encode_image(image_paths[0])
    base64_image2 = resize_and_encode_image(image_paths[1])
    base64_image3 = resize_and_encode_image(image_paths[2])
    base64_image4 = resize_and_encode_image(image_paths[3])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}",
            },
        },
        {
            "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}",
            },
        },
        {
            "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image3}",
            },
        },
        {
            "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image4}",
            },
        },
        ]
        }
        ],
            "max_tokens": 1200
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    with open(image_paths[0].replace('png','json'),'w') as jsonf:
        json.dump(response.json(),jsonf)


if __name__=='__main__':
    with open('chatgpt_4_v.txt','r') as lf:
        all_labels = lf.readlines()
    for lfile in all_labels:
        lfile = lfile.strip()
        color_img_path00 = f"{lfile[:-4]}_patch00.png"
        color_img_path01 = f"{lfile[:-4]}_patch01.png"
        color_img_path10 = f"{lfile[:-4]}_patch10.png"
        color_img_path11 = f"{lfile[:-4]}_patch11.png"

        im00 = np.array(Image.open(color_img_path00))
        im01 = np.array(Image.open(color_img_path01))
        im10 = np.array(Image.open(color_img_path10))
        im11 = np.array(Image.open(color_img_path11))
        seg_map_00 = convert_color_map_to_segmentation(im00, colors)
        seg_map_01 = convert_color_map_to_segmentation(im01, colors)
        seg_map_10 = convert_color_map_to_segmentation(im10, colors)
        seg_map_11 = convert_color_map_to_segmentation(im11, colors)
        segmentation_maps = [seg_map_00,seg_map_01,seg_map_10,seg_map_11]
        image_paths = [color_img_path00,color_img_path01,color_img_path10,color_img_path11]
        generate_captions(image_paths, segmentation_maps)
        time.sleep(20)

