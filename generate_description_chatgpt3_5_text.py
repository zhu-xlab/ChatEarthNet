# analyze the image and generate the corresponding prompts

import numpy as np
from PIL import Image
import base64
import requests
import io
import json
import time
import random
import functools
import operator
import copy

Threshold = 20
from requests.exceptions import RequestException, Timeout


land_cover_label = {'land_cover': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'background',
            'water', 'developed area', 'tree', 'shrub', 'grass',
            'crop', 'bare land', 'snow', 'wetland', 'mangroves', 'moss',
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


import numpy as np
from PIL import Image
from collections import Counter

def divide_into_patches(image):
    rows, cols = image.shape
    return {
        "top left": image[:rows//2, :cols//2],
        "top right": image[:rows//2, cols//2:],
        "bottom left": image[rows//2:, :cols//2],
        "bottom right": image[rows//2:, cols//2:],
        "middle": image[rows//4:3*rows//4, cols//4:3*cols//4]
    }


def count_pixel_proportions1(num, land_type, patch):
    count = np.count_nonzero(patch == land_type)
    proportions = count / (256.*256.)
    return proportions

def overall_distribution(image):
    all_land_types = Counter(list(image.reshape(-1)))
    analyze_prompt = ""
    all_land_types = dict(sorted(all_land_types.items(), key=lambda item: item[1], reverse=True))

    for land_type, num in all_land_types.items():
        #print(f"{labels[land_type]}:{all_land_types[land_type]}")
        if land_type == 0 or all_land_types[land_type]<Threshold:
            continue
        analyze_prompt += f"{labels[land_type]}; "
    return analyze_prompt


def find_most_frequent_types_revised(patch):
    patch = functools.reduce(operator.iconcat, patch, [])
    patch = [it for it in patch if it!=0]
    values, counts = np.unique(patch, return_counts=True)
    dvalues = copy.deepcopy(values)
    dcounts = copy.deepcopy(counts)

    sumall = sum(dcounts)
    count_dict = {} 
    for i in range(len(dvalues)):
        count_dict[dvalues[i]] = dcounts[i] / float(sumall)
    
    ncounts = []
    nvalues = []
    for i in range(len(counts)):
        if counts[i]>=Threshold:
            ncounts.append(counts[i])
            nvalues.append(values[i])
    
    # Combine values and counts and sort by frequency in descending order
    frequencies = np.column_stack((nvalues, ncounts))
    frequencies = frequencies[frequencies[:, 1].argsort()[::-1]]  # Sort by frequency
    # Return the three most frequent values or all if less than three
    out_idx = frequencies[:3, 0] if frequencies.shape[0] >= 3 else frequencies[:, 0]
    return out_idx, count_dict

def convert_percent_range(percent):
    candidates = ['fraction', 'part', 'portion', 'amount', 'quantity']
    if percent>=0 and percent<=9:
        return f'extra small {random.choice(candidates)}'
    elif percent>=10 and percent<=19:
        return f'small {random.choice(candidates)}'
    elif percent>=20 and percent<=49:
        return f'medium {random.choice(candidates)}'
    elif percent>=50 and percent<=79:
        return f'large {random.choice(candidates)}'
    elif percent>=80 and percent<=100:
        return f'extra large {random.choice(candidates)}'
    else:
        raise Exception('Wrong portion')


def post_with_retry(url, headers=None, json=None, max_retries=3, delay=2):
    """
    Makes a POST request to a specified URL with a retry mechanism.
    
    :param url: URL to which the POST request is made
    :param data: Dictionary, list of tuples, bytes, or file-like object to send in the body
    :param headers: Dictionary of HTTP Headers to send with the request
    :param max_retries: Maximum number of retries
    :param delay: Delay between retries in seconds
    :return: Response object
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json, timeout=60)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            return response
        except Timeout:
            time.sleep(delay)
        except RequestException as e:
            print(f"Request failed: {e}. Attempt {attempt + 1} of {max_retries}. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    raise Exception(f"Failed to POST to {url} after {max_retries} retries")


def analyze_segmentation_map(image):
    overalld = overall_distribution(image)
    # Divide the segmentation map into patches
    patches = divide_into_patches(image)
    # Find three largest infrequent pixel types for each patch
    #frequent_types = {patch_name: find_most_frequent_types_revised(patch) for patch_name, patch in patches.items()}
    frequent_types = {}
    frequent_percents = {}
    percents = {}
    for patch_name, patch in patches.items():
        out_idx, cdict = find_most_frequent_types_revised(patch)
        out_percent = [convert_percent_range(int(round(cdict[it],2)*100)) for it in out_idx]

        frequent_types[patch_name] = out_idx
        frequent_percents[patch_name] = out_percent

    context_str = overalld + "\n"
    for patch_name, lst in frequent_types.items():
        frequent_percent = frequent_percents[patch_name]
        print(frequent_percent)
        context_str += f"The {patch_name} mainly contains the following land cover types, in descending order of content: "
        if len(lst)==3:
            context_str += f"{labels[lst[0]]} ({frequent_percent[0]}), {labels[lst[1]]} ({frequent_percent[1]}), and {labels[lst[2]]} ({frequent_percent[2]}).\n"
        if len(lst)==2:
            context_str += f"{labels[lst[0]]} ({frequent_percent[0]}) and {labels[lst[1]]} ({frequent_percent[1]}).\n"
        if len(lst)==1:
            context_str += f"{labels[lst[0]]} ({frequent_percent[0]}).\n"

    return context_str


def generate_captions(image_paths, segmentation_maps):
    user_prompts = []
    for image in segmentation_maps:
        analyze_prompt = analyze_segmentation_map(image)
        user_prompt = "Analyze the provided image as an AI visual assistant. The following contexts are provided.\n"
        user_prompt += "The overall land cover distributions from most to least are: "
        user_prompt += analyze_prompt
        user_prompt += '\n'
        user_prompts.append(user_prompt)


    system_prompt = "You are an AI visual assistant who can help describe images based on the given contexts. Please write the description in a paragraph, and avoid saying other things. The following constraints should be obeyed:\n\
        1) Describe the image in the order of the spatial distributions presented in the given contexts. Link descriptions of different parts to make the overall image description more fluent.\
        2) Describe the dominant land cover type in the image and its spatial locations.\
        3) Describe the land cover types in each part of the image in descending order of their coverage areas.\
        4) Diversify descriptions related to portions in each paragraph. \
        5) Summarize the main theme of the image in the final sentence.\
        6) Describe it objectively; do not use words: 'possibly', 'likely', 'perhaps', 'context', 'segmentation', 'appear', 'change', 'transition', 'dynamic', or any words with similar connotations."


    api_key = "openai api key"

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    model = "gpt-3.5-turbo-1106"

    data1 = {
        "model": model,
        "max_tokens": 300,
        "messages": [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompts[0]
            },
        ]
    }
    data2 = {
        "model": model,
        "max_tokens": 300,
        "messages": [
            {
                'role': 'system',
                'content': system_prompt,
            },

            {
                'role': 'user',
                'content': user_prompts[1]
            },
        ]
    }
    data3 = {
        "model": model,
        "max_tokens": 300,
        "messages": [
            {
                'role': 'system',
                'content': system_prompt,
            },

            {
                'role': 'user',
                'content': user_prompts[2]
            },
        ]
    }
    data4 = {
        "model": model,
        "max_tokens": 300,
        "messages": [
            {
                'role': 'system',
                'content': system_prompt,
            },

            {
                'role': 'user',
                'content': user_prompts[3]
            },
        ]
    }


    response = post_with_retry(url, headers=headers, json=data1)
    time.sleep(0.5)
    response2 = post_with_retry(url, headers=headers, json=data2)
    time.sleep(0.5)
    response3 = post_with_retry(url, headers=headers, json=data3)
    time.sleep(0.5)
    response4 = post_with_retry(url, headers=headers, json=data4)
    time.sleep(0.5)
    responses = [response,response2,response3,response4]
    for i,response in enumerate(responses):
        print('-----------------------------------------------------------')
        print(response.json())
        with open(image_paths[i].replace('.png','_chatgpt_3_5.json'),'w') as jsonf:
            json.dump(response.json(),jsonf)
        print('-----------------------------------------------------------')


if __name__=='__main__':
    with open('chatgpt_3_5_label_file_path.txt','r') as lf:
        all_labels = lf.readlines()
    selects = all_labels
    select_3000 = selects

    for lfile in label_list:
        # Example usage
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

