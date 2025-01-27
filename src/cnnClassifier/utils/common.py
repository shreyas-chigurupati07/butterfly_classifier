import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a yaml file and return a ConfigBox object

    Args:
        path_to_yaml: Path to the yaml file
    Raises:
        BoxValueError: If the yaml file cannot be read
        e: empty file
    Returns:
        ConfigBox: A ConfigBox object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'YAML file loaded successfully: {path_to_yaml}')
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_dir(path_to_dir: list, verbose=True):
    """
    Create a directory if it does not exist

    Args:
        path_to_dir: Path to the directory
        verbose: Print the message
    Raises:
        e: If the directory cannot be created
    """
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Directory created successfully: {path}')


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save a dictionary to a json file

    Args:
        path (Path): Path to the json file
        data (dict): data to be saved inside the json file
    """

    with open(path, 'w') as f:
        json.dump(data, f, indent=1)
    logger.info(f'JSON file saved successfully: {path}')


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a json file and return a ConfigBox object
    
    Args:
        path (Path): Path to the json file
    Returns:
        ConfigBox: A ConfigBox object
    
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f'JSON file loaded successfully from path: {path}')
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save a binary file

    Args:
        data (Any): Data to be saved
        path (Path): Path to the binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f'Binary file saved successfully at: {path}')


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file

    Args:
        path (Path): Path to the binary file
    Returns:
        Any: Data loaded from the binary file
    """
    data = joblib.load(path)
    logger.info(f'Binary file loaded successfully from path: {path}')
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in KB

    Args:
        path (Path): Path to the file
    Returns:
        str: Size of the file
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    logger.info(f'Size of the file: {path} is {size_in_kb} bytes')
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring, filename):
    """
    Decode the image from base64 string and save it to the disk

    Args:
        imgstring (str): Base64 string of the image
        filename (str): Name of the file
    """
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    """
    Encode the image into base64 string

    Args:
        croppedImagePath (str): Path to the image
    Returns:
        str: Base64 string of the image
    """
    with open(croppedImagePath, "rb") as img_file:
        return base64.b64encode(img_file.read())
