def detect_backend() -> str:
    """
    Detects whether to use Numpy or Tensor based on GPU availability

    Returns:
        - "numpy": If no GPU is available
        - "tensor": If PyTorch is installed and GPU is available
        - "tensorflow": If TensorFlow is installed and GPU is available
    """
    try:
        # Check for TensorFlow Support
        import tensorflow as tf
        if len(tf.config.list_physical_devices("GPU")) > 0:
            return "tensorflow"
    except ImportError:
        pass

    try:
        # Check for PyTorch Support
        import torch
        if torch.cuda.is_available():
            return "tensor"
    except ImportError:
        pass

    # The default is numpy (no GPU)
    return "numpy"
