import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="text-disentangling-vae-jvasilakes",
    version="0.0.1",
    author="Jake Vasilakes",
    author_email="jake.vasilakes@manchester.ac.uk",
    description="VAE for disentangled text representations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages() + ["vae"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
