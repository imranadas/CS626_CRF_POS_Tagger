import torch
import tensorflow as tf
import tensorrt as trt
import onnx
import sys

# Function to print a message if a library is unavailable
def check_library(lib_name):
    print(f"\nChecking {lib_name}:")

# 1. Check PyTorch GPU support
check_library("PyTorch")
if torch.cuda.is_available():
    print(f"PyTorch - CUDA available: True")
    print(f"PyTorch - CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"PyTorch - Device count: {torch.cuda.device_count()}")
else:
    print("PyTorch - No GPU available.")

# 2. Check TensorFlow GPU support
check_library("TensorFlow")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow - GPU available: True")
    for gpu in gpus:
        print(f"TensorFlow - GPU name: {gpu.name}")
else:
    print("TensorFlow - No GPU available.")

# 3. Check TensorRT GPU support
check_library("TensorRT")
try:
    print(f"TensorRT version: {trt.__version__}")
    # Note: TensorRT typically doesn't have a direct GPU check in Python.
    # It is assumed if installed with GPU support that it will use it.
except Exception as e:
    print(f"TensorRT - Error: {e}")

# 4. Check ONNX support
check_library("ONNX")
try:
    print(f"ONNX version: {onnx.__version__}")
    # ONNX doesn't directly manage GPUs but can work with GPU-based runtimes.
    # Add here additional checks like loading ONNXRuntime with GPU support if needed.
except Exception as e:
    print(f"ONNX - Error: {e}")

print("\nTest Complete.")
