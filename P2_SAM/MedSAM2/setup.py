"""
setup.py
========
Installation script for the HECKTOR-MedSAM2 project.
"""

import os
from setuptools import find_packages, setup

NAME = "hecktor_medsam2"
VERSION = "1.0.0"
DESCRIPTION = (
    "MedSAM2 fine-tuned for HECKTOR head-and-neck tumour segmentation "
    "(GTVp + GTVn) from dual-modality CT + PET data."
)

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

REQUIRED = [
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "numpy>=2.0.1",
    "tqdm>=4.66.5",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=10.4.0",
    "SimpleITK>=2.4.0",
    "huggingface-hub>=0.20.0",
    "tensordict>=0.5.0",
    "fvcore>=0.1.5.post20221221",
    "pandas>=2.2.0",
    "scikit-image>=0.24.0",
    "tensorboard>=2.17.0",
    "matplotlib>=3.9.0",
]

EXTRAS = {
    "train": [
        "submitit>=1.5.1",
        "omegaconf>=2.3.0",
    ],
    "dev": [
        "pytest>=8.0.0",
        "black>=24.0.0",
    ],
}

BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_ERROR_MSG = (
    "{}\n\nFailed to build the SAM2 CUDA extension. "
    "You can still use MedSAM2 without it (some post-processing steps will "
    "be skipped).\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []
    try:
        from torch.utils.cpp_extension import CUDAExtension
        srcs = ["sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        return [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            return []
        raise


try:
    from torch.utils.cpp_extension import BuildExtension

    class BuildExtensionIgnoreErrors(BuildExtension):
        def finalize_options(self):
            try:
                super().finalize_options()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def get_ext_filename(self, ext_name):
            try:
                return super().get_ext_filename(ext_name)
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []
                return "_C.so"

    cmdclass = {
        "build_ext": (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        )
    }
except Exception as e:
    cmdclass = {}
    if BUILD_ALLOW_ERRORS:
        print(CUDA_ERROR_MSG.format(e))
    else:
        raise

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Ethan",
    python_requires=">=3.10",
    packages=find_packages(exclude=["notebooks", "scripts"]),
    include_package_data=True,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
