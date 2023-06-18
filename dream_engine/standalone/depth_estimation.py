import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

"""
Use 'justinsoberano/depth-ai' for general image depth analysis
Use 'justinsoberano/rock-depth-ai for up-close/macro image depth analysis
depth-ai provides a larger depth threshold, allowing for far-away objects to be detected
rock-depth-ai has a smaller depth threshold, but allows for greater detail on upclose objects

!Does not work on transparent images!
"""

extractor = GLPNImageProcessor.from_pretrained("justinsoberano/depth-ai")
model = GLPNForDepthEstimation.from_pretrained("justinsoberano/depth-ai")

"""
Quality: [1, 5]
Any higher value will cause the program to become unstable and may terminate.
Wait time exponentially increases the higher the quality level.
"""
quality = 4

def predict_depth(image):

    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width * quality, new_height * quality)
    image = image.resize(new_size)

    inputs = extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    pad = 16
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    return image, output

def predict(image):
    print('Starting depth analysis.')
    image, depth_map = predict_depth(image)
    depth_image = (depth_map * 255 / np.max(depth_map)).astype('uint8')
    
    # Colormap can be changed to different colors, 
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colormap = plt.get_cmap('Greys')
    depth_image = (colormap(depth_image) * 255).astype('uint8')
    depth_image = Image.fromarray(depth_image)
    return depth_image

def cls():
    os.system('cls' if os.name=='nt' else 'clear')
    

def init():
    cls()
    name = input("Enter image name for depth prediction: ")
    img = Image.open(os.path.join('/images/' + name))
    image = predict(img)
    print('Depth map exported.\n')
    image.save('predicted_depth.png')
    
# cls()
# image = predict(img)
# print('Depth map exported.\n')
# image.save('predicted_depth.png')