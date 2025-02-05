from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
import os
os.path.dirname(os.path.abspath(__file__))

EIGEN_INCLUDE_DIR = "/usr/include/eigen3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="gsaligner",
    packages=["gsaligner"],
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="gsaligner._C",
            sources=[
                "gsaligner/aligner.cu",
                "gsaligner/aligner_utils.cu",
                "ext.cpp",
                "gsaligner/tools/image_set.cpp",
                "gsaligner/tools/image_set_impl.cu",
                "gsaligner/tools/mad_tree.cpp",
                "gsaligner/tools/sum_reduce.cu",
                "gsaligner/registration/geometric.cpp",
                "gsaligner/registration/photometric.cpp",
                "gsaligner/registration/photometric_cuda.cu",
                "gsaligner/registration/vel_estimator.cpp",

            ],
            include_dirs=[
                EIGEN_INCLUDE_DIR,
                os.path.join(BASE_DIR, "gsaligner"),
                os.path.join(BASE_DIR, "gsaligner/tools"),
                os.path.join(BASE_DIR, "gsaligner/registration"),],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
