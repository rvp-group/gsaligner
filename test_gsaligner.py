import matplotlib.pyplot as plt
import torch
import numpy as np
from gsaligner._C import GSAligner
import open3d as o3d
from pathlib import Path
torch.manual_seed(42)
BASE_PATH = Path("/home/rvp-00/Desktop/gsaligner_data")

ref_pcd = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_ref.ply"))
# query_pcd = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_query.ply"))
query_pcd = o3d.io.read_point_cloud(str(BASE_PATH / "cloud_ref.ply"))
ref_depth_np = np.load(BASE_PATH / "depth_ref.npy")
# query_depth_np = np.load(BASE_PATH / "depth_query.npy")
query_depth_np = np.load(BASE_PATH / "depth_ref.npy")
ref_projmat_np = np.load(BASE_PATH / "K_ref.npy")
# Initialize with the same value for now
query_projmat_np = np.load(BASE_PATH / "K_ref.npy")

H, W = 64, 1024
initial_guess = torch.eye(4).double()

ref_depth = torch.from_numpy(ref_depth_np).float().cuda()[None, ...]
query_depth = torch.from_numpy(query_depth_np).float().cuda()[None, ...]
ref_cloud = torch.from_numpy(np.asarray(ref_pcd.points)).double()
query_cloud = torch.from_numpy(np.asarray(query_pcd.points)).double()
ref_projmat = torch.eye(4).float().cuda()
query_projmat = torch.eye(4).float().cuda()
ref_projmat[:3, :3] = torch.from_numpy(ref_projmat_np).float().cuda()
query_projmat[:3, :3] = torch.from_numpy(query_projmat_np).float().cuda()

ref_projmat = ref_projmat.transpose(0, 1)
query_projmat = query_projmat.transpose(0, 1)

aligner = GSAligner(
    H,
    W,
    0.1,
    0.2,
    0.02,
    0.8,
    10,
    5.0,
    0.8,
    1.0,
    0.5,
    80,
    10
)

aligner.setReference(ref_depth, ref_cloud, ref_projmat)
aligner.setQuery(query_depth, query_cloud, query_projmat)
T_geom, fitness_geom, rmse_geom = aligner.alignGeometric(torch.eye(4).double())
# print(T_geom.cpu().numpy(), fitness_geom, rmse_geom)
aligner.alignPhotometric(T_geom.float())
# fig, axs = plt.subplots(2, 1)
# axs[0].imshow(ref_depth_np, vmin=0, vmax=10)
# axs[1].imshow(query_depth_np, vmin=0, vmax=10)
# plt.show()

# res = RegistrationGeomPhoto(
#     initial_guess,
#     ref_depth,
#     query_depth,
#     ref_cloud,
#     query_cloud,
#     ref_projmat,
#     H,
#     W,
#     0.1,
#     0.2,
#     0.02,
#     0.8, 10, 0, 0, 0, 0, 0, 0)
