import setuptools

setuptools.setup(
    name="tsml_neuro",
    version="1.0",
    packages=setuptools.find_packages(),
    install_requires = ["sktime","mne","pyprep"]
)
