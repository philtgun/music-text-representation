from setuptools import setup

setup(
    name="mtr",
    packages=["mtr"],
    install_requires=[
        "librosa >= 0.8",
        "torchaudio_augmentations >= 0.2.1",  # for augmentation
        "torch",
        "numpy",
        "pandas",
        "einops",
        "scikit-learn",
        "wandb",
        "jupyter",
        "matplotlib",
        "omegaconf",
        "astropy",
        "transformers",
    ],
)
