import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import typer
from typing_extensions import Annotated
import time

# # binded vectors and madtree
from gsaligner.src.pybind.pyvector import VectorEigen3d
from gsaligner.src.pybind.pygsaligner import GSAligner

from copy import deepcopy

MAX_ITERATIONS = 15
app = typer.Typer()
T_guess = np.eye(4)

PATH_TO_TEST_DATA = "/home/ldg/source/rvp/gsaligner/test_data/gsaligner_data"

def main(viz: Annotated[bool, typer.Option(help="if true visualizer on", show_default=True)] = False) -> None:
    calib_mat = np.load(os.path.join(PATH_TO_TEST_DATA, "K_ref.npy"))
    cloud_query = o3d.io.read_point_cloud(os.path.join(PATH_TO_TEST_DATA, "cloud_query.ply"))
    cloud_ref = o3d.io.read_point_cloud(os.path.join(PATH_TO_TEST_DATA, "cloud_ref.ply"))
    depth_query = np.load(os.path.join(PATH_TO_TEST_DATA, "depth_query.npy"))
    depth_ref = np.load(os.path.join(PATH_TO_TEST_DATA, "depth_ref.npy"))
    # depth_ref = depth_query.copy()

    # aligner = GSAligner(num_threads=os.cpu_count())
    # aligner = GSAligner(num_threads=1)
    # aligner.setQueryCloud(VectorEigen3d(np.asarray(cloud_query.points)))
    # aligner.setReferenceCloud(VectorEigen3d(np.asarray(cloud_ref.points)))
    # aligner.setImages(depth_query, depth_ref)
    # aligner.setReferenceCameraMatrix(calib_mat)
    
    # T_guess[:3, :3] = R.from_euler('xyz', [0.1, 0.1, 0.1]).as_matrix()
    # T_guess[:3, 3] = np.random.rand(3)
    # print("init guess T\n", T_guess)
    # print("gt T\n", np.eye(4))
    
    # T_est = aligner.compute(np.eye(4), icp_iterations=MAX_ITERATIONS)
    # print("est T\n", T_est)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        cloud_ref, cloud_query, 0.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(reg_p2p)
    print("p2p transform")
    print(reg_p2p.transformation)
    T_est = np.linalg.inv(reg_p2p.transformation)


    cloud_query_in_ref = deepcopy(cloud_query).transform(T_est)
    cloud_ref.paint_uniform_color(np.float32([1, 0, 0]))
    cloud_query_in_ref.paint_uniform_color(np.float32([0, 1, 0]))
    cloud_query.paint_uniform_color(np.float32([0.3, 0.3, 0.3]))

    # o3d.visualization.draw_geometries([cloud_ref, cloud_query_in_ref, cloud_query])
    o3d.visualization.draw_geometries([cloud_ref, cloud_query_in_ref])  


def run():
    typer.run(main)

if __name__ == '__main__':
    run()