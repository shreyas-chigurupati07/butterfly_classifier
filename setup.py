import setuptools
import os

# with open("README.md", "r") as fh:
#     long_description = fh.read()
readme_file = "README.md" if os.path.exists("README.md") else None
long_description = open(readme_file, "r").read() if readme_file else "A small package to classify butterflies"


__version__ = '0.1.0'


REPO_NAME = 'butterfly_classifier'
AUTHOR_USERNAME = 'shreyas-chigurupati07'
SRC_REPO = 'cnnClassifier'
AUTHOR_EMAIL = 'chigurupatishreyas@gmail.com'


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USERNAME,
    author_email=AUTHOR_EMAIL,
    description="A small package to classify butterflies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),

)


