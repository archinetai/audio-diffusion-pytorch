from setuptools import find_packages, setup

setup(
    name="audio-diffusion-pytorch",
    packages=find_packages(exclude=[]),
    version="0.1.1",
    license="MIT",
    description="Audio Diffusion - PyTorch",
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/audio-diffusion-pytorch",
    keywords=["artificial intelligence", "deep learning", "audio generation"],
    install_requires=[
        "tqdm",
        "torch>=1.6",
        "torchaudio",
        "data-science-types>=0.2",
        "einops>=0.6",
        "a-unet",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
