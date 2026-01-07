import tensorflow as tf
import os

print("=" * 60)
print("TensorFlow & GPU Environment Check")
print("=" * 60)

# TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# CUDA / cuDNN version (from TF build info)
build_info = tf.sysconfig.get_build_info()
cuda_version = build_info.get("cuda_version", "Not available")
cudnn_version = build_info.get("cudnn_version", "Not available")

print(f"CUDA version (TF built with): {cuda_version}")
print(f"cuDNN version (TF built with): {cudnn_version}")

# List physical devices
gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

print("\nDetected devices:")
print(f"  CPUs: {len(cpus)}")
print(f"  GPUs: {len(gpus)}")

if not gpus:
    print("❌ No GPU detected. TensorFlow will use CPU.")
    print("   Please check CUDA / cuDNN / driver compatibility.")
else:
    print("✅ GPU(s) detected:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu}")

# Optional: show logical devices
logical_gpus = tf.config.list_logical_devices("GPU")
if logical_gpus:
    print(f"\nLogical GPUs: {len(logical_gpus)}")

# Test GPU computation
print("\nTesting GPU computation...")

device_name = "/GPU:0" if gpus else "/CPU:0"
with tf.device(device_name):
    a = tf.random.normal([5000, 5000])
    b = tf.random.normal([5000, 5000])
    c = tf.matmul(a, b)

print(f"✅ Computation finished on {device_name}")
print(f"Output tensor shape: {c.shape}")

print("=" * 60)