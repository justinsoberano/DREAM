from PIL import Image
import torch
from tqdm.auto import tqdm
import os
import open3d as o3d
import numpy as np

from rembg import remove
from dream_engine.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from dream_engine.diffusion.sampler import PointCloudSampler
from dream_engine.models.download import load_checkpoint
from dream_engine.models.configs import MODEL_CONFIGS, model_from_config
from dream_engine.util.plotting import plot_point_cloud

def cls():
    os.system('cls' if os.name=='nt' else 'clear')
cls()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DREAM - Depth Rendering Engine for Automated Modeling\n")

"""
Render times are SIGNIFICANTLY reduced when using an NVIDIA GPU.
CPU used: Apple M1 Max, 32GB

Render Quality:
base40M, low <- About 15 minutes to render on CPU
base300M, mid <- About 1 hour to render on CPU (GPU with CUDA recommended)
base1B, high <- About 15 minutes to render on CUDA (CPU is EXTREMELY slow)
"""

engine = "base40M"

base_model = model_from_config(MODEL_CONFIGS[engine], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[engine])
print("Generating 3D point cloud...")

upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

base_model.load_state_dict(load_checkpoint(engine, device))
upsampler_model.load_state_dict(load_checkpoint("upsample", device))

sampler = PointCloudSampler(
    device = device,
    models = [base_model, upsampler_model],
    diffusions = [base_diffusion, upsampler_diffusion],
    num_points = [1024, 4096 - 1024],
    aux_channels=["R", "G", "B"],
    guidance_scale=[3.0, 3.0],
)

img = Image.open(os.path.join('./rem_bg.png'))
img_opt = remove(img);

samples = None

for x in tqdm(sampler.sample_batch_progressive(batch_size = 1, model_kwargs = dict(images=[img_opt]))):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]

with open('pc.ply', 'wb') as f:
    pc.write_ply(f)
print("Point cloud exported.")

fig = plot_point_cloud(pc, grid_size = 3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

pcd = o3d.io.read_point_cloud('pc.ply')
pcd.estimate_normals()
pcd.orient_normals_towards_camera_location(pcd.get_center())
pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
o3d.io.write_triangle_mesh('mesh.ply', mesh)
print("Mesh created.\n")