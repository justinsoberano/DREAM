"""
Stand-alone mesh creator. Used for testing.
"""

import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud('pc.ply')
pcd.estimate_normals()
pcd.orient_normals_towards_camera_location(pcd.get_center())
pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
o3d.io.write_triangle_mesh('mesh.ply', mesh)

# Add functionality to clean up generated models and remove outliers
# http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Connected-components

