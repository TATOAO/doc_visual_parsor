# Doc Visual Parser

A lightweight document layout detection library using ONNX models and PyMuPDF.

## Installation

### CPU-Only Installation (Default)
For users without GPU or who prefer CPU-only inference:

```bash
pip install doc-chunking
```

### GPU Installation (CUDA Support)
For users with NVIDIA GPUs who want accelerated inference:

```bash
pip install doc-chunking[gpu]
```

**Note**: GPU installation requires:
- NVIDIA GPU with CUDA support
- Compatible CUDA drivers installed
- Compatible cuDNN version

### Development Installation
For development with all dependencies:

```bash
pip install doc-chunking[all]
```

## CUDA Requirements

If you're installing the GPU version, ensure you have:

1. **NVIDIA GPU** with CUDA Compute Capability 6.0 or higher
2. **CUDA Toolkit** (version 11.8 or 12.x recommended)
3. **cuDNN** (compatible with your CUDA version)

### CUDA Version Compatibility

| ONNX Runtime GPU | CUDA Version | cuDNN Version |
|------------------|--------------|---------------|
| 1.15.0+         | 11.8, 12.x   | 8.6+          |

### Installation Verification

After installation, you can verify CUDA support:

```python
import onnxruntime as ort

# Check available providers
print("Available providers:", ort.get_available_providers())

# Should include 'CUDAExecutionProvider' if GPU support is working
```

## Usage

The library automatically detects and uses the best available execution provider:

```python
from doc_chunking import ONNXLayoutDetector

# Will automatically use CUDA if available, otherwise CPU
detector = ONNXLayoutDetector(
    model_path="path/to/model.onnx",
    device="auto"  # or "cuda" to force GPU, "cpu" to force CPU
)

# For advanced GPU configuration
detector = ONNXLayoutDetector(
    model_path="path/to/model.onnx",
    device="cuda",
    cuda_device_id=0,  # Use specific GPU
    gpu_mem_limit=4 * 1024**3  # Limit GPU memory to 4GB
)
```

## Troubleshooting

### CUDA Issues
- Ensure CUDA drivers are properly installed: `nvidia-smi`
- Verify CUDA version compatibility with ONNX Runtime
- Check that `CUDAExecutionProvider` appears in available providers

### Performance Tips
- Use `device="auto"` for automatic provider selection
- Set `gpu_mem_limit` to prevent GPU memory issues
- Use `cuda_device_id` to specify which GPU to use in multi-GPU systems

## License

MIT License - see LICENSE file for details.
