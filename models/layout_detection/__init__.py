# Acknowledgement: https://github.com/opendatalab/DocLayout-YOLO
# DocLayout-YOLO is a model for detecting the layout of a document.
# It is a YOLO-based model that is trained on a dataset of document images.
# It is used to detect the layout of a document, such as the presence of a table, a list, or a text block.

from .detector import DocLayoutDetector
from .download_model import download_model, list_available_models, MODELS

__all__ = ['DocLayoutDetector', 'download_model', 'list_available_models', 'MODELS']



