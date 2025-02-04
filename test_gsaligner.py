import torch
import numpy as np
from gsaligner._C import RegistrationGeomPhoto
import open3d as o3d
from pathlib import Path
torch.manual_seed(42)
BASE_PATH = Path("/home/rvp-00/Desktop/gsaligner_data")

ref_pcd = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_ref.ply"))
query_pcd = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_query.ply"))


H, W = 128, 1024
initial_guess = torch.eye(4).float().cuda()

ref_depth = torch.rand((1, H, W)).float().cuda()
query_depth = torch.rand((1, H, W)).float().cuda()
ref_cloud = torch.from_numpy(np.asarray(ref_pcd.points)).float()
query_cloud = torch.from_numpy(np.asarray(query_pcd.points)).float()
# ref_cloud = torch.rand((H*W, 3)).float().cpu()
# query_cloud = torch.rand((H*W, 3)).float().cpu()
ref_projmat = torch.eye(4).float().cuda()


res = RegistrationGeomPhoto(
    initial_guess,
    ref_depth,
    query_depth,
    ref_cloud,
    query_cloud,
    ref_projmat,
    H,
    W,
    0.1,
    0.2,
    0.02,
    0.8, 10, 0, 0, 0, 0, 0, 0)
