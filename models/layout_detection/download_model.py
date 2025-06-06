"""
Download and manage DocLayout-YOLO models from Hugging Face.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "docstructbench": {
        "repo_id": "juliozhao/DocLayout-YOLO-DocStructBench",
        "filename": "doclayout_yolo_docstructbench_imgsz1024.pt",
        "description": "Fine-tuned on DocStructBench for various document types"
    },
    "d4la": {
        "repo_id": "juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained",
        "filename": "best.pt",
        "description": "D4LA dataset trained model"
    },
    "doclaynet": {
        "repo_id": "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained", 
        "filename": "best.pt",
        "description": "DocLayNet dataset trained model"
    }
}

def get_model_cache_dir():
    """Get the model cache directory."""
    cache_dir = Path.home() / ".cache" / "doclayout_yolo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_model(model_name: str = "docstructbench", force_download: bool = False):
    """
    Download a DocLayout-YOLO model from Hugging Face.
    
    Args:
        model_name: Name of the model to download ('docstructbench', 'd4la', 'doclaynet')
        force_download: Whether to force re-download even if model exists
        
    Returns:
        Path to the downloaded model file
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODELS.keys())}")
    
    model_config = MODELS[model_name]
    cache_dir = get_model_cache_dir()
    local_path = cache_dir / f"{model_name}_{model_config['filename']}"
    
    if local_path.exists() and not force_download:
        logger.info(f"Model {model_name} already exists at {local_path}")
        return str(local_path)
    
    logger.info(f"Downloading {model_name} model: {model_config['description']}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_config["repo_id"],
            filename=model_config["filename"],
            cache_dir=str(cache_dir),
            force_download=force_download
        )
        
        # Create a symlink or copy to a predictable location
        if not local_path.exists():
            import shutil
            shutil.copy2(downloaded_path, local_path)
        
        logger.info(f"Successfully downloaded {model_name} model to {local_path}")
        return str(local_path)
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        raise

def list_available_models():
    """List all available models and their descriptions."""
    print("Available DocLayout-YOLO models:")
    for name, config in MODELS.items():
        cache_dir = get_model_cache_dir()
        local_path = cache_dir / f"{name}_{config['filename']}"
        status = "✓ Downloaded" if local_path.exists() else "✗ Not downloaded"
        print(f"  {name}: {config['description']} [{status}]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DocLayout-YOLO models")
    parser.add_argument("--model", default="docstructbench", 
                       choices=list(MODELS.keys()),
                       help="Model to download")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if model exists")
    parser.add_argument("--list", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    else:
        download_model(args.model, args.force)
