
"""
Test version of depth.py
This creates a visual 3D mesh based on the predicted depth.
Used for model training and prediction fine tuning.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import torch
import tempfile
import os
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

extractor = GLPNImageProcessor.from_pretrained("justinsoberano/depth-ai")
model = GLPNForDepthEstimation.from_pretrained("justinsoberano/depth-ai")
img = Image.open(os.path.join('Mesh Creator', './rock.png'))

def predict_depth(image):
    
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width * 2, new_height * 2)
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

def generate_mesh(image, depth_image):
    width, height = image.size

    # depth_image = (depth_map * 255 / np.max(depth_map)).astype('uint8')
    image = np.array(image)

    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                    convert_rgb_to_intensity=False)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)

    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction(orientation_reference=(0., 0., -1.))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, np.pi, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    temp_name = next(tempfile._get_candidate_names()) + '.obj'
    o3d.io.write_triangle_mesh(temp_name, mesh)

    return temp_name

def predict(image):
    image, depth_map = predict_depth(image)
    depth_image = (depth_map * 255 / np.max(depth_map)).astype('uint8')
    mesh_path = generate_mesh(image, depth_image)
    colormap = plt.get_cmap('plasma')
    depth_image = (colormap(depth_image) * 255).astype('uint8')
    depth_image = Image.fromarray(depth_image)
    
    depth_image.save('depth.png')

    return depth_image, mesh_path

predict(img)