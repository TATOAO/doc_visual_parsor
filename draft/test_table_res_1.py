import onnxruntime as ort
import cv2
import numpy as np

model_path = "/Users/tatoao_mini/Work/doc_visual_parsor/model_parameters/layout_detection/OpenDataLab/PDF-Extract-Kit-1___0/models/TabRec/SlanetPlus/slanet-plus.onnx"
image_path = "table_demo1_withline.png"
# image_path = "/Users/tatoao_mini/Work/doc_visual_parsor/demo_table_wo_line.png"
session = ort.InferenceSession(model_path)

# Get input and output information
input_info = session.get_inputs()[0]
all_outputs = session.get_outputs()

input_name = input_info.name
output_names = [output.name for output in all_outputs]

print(f"Input name: {input_name}")
print(f"Input shape: {input_info.shape}")
print(f"Input type: {input_info.type}")

print(f"Number of outputs: {len(all_outputs)}")
for i, output in enumerate(all_outputs):
    print(f"Output {i}: name={output.name}, shape={output.shape}, type={output.type}")

print(f"Available providers: {session.get_providers()}")

# Load and preprocess image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image")
    exit(1)

print(f"Original image shape: {image.shape}")

# Preprocessing for SlanetPlus table structure recognition
# Model expects dynamic dimensions but typically works with 1024x1024
input_size = 1024  # Use 1024 as default size for SlanetPlus

original_height, original_width = image.shape[:2]
print(f"Original image size: {original_width}x{original_height}")

# Resize image while maintaining aspect ratio
scale = min(input_size / original_width, input_size / original_height)
new_width = int(original_width * scale)
new_height = int(original_height * scale)

print(f"Scale factor: {scale:.3f}")
print(f"Resized dimensions: {new_width}x{new_height}")

# Resize image
resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Create padded image with padding value 114 (commonly used in table models)
padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
padded[:new_height, :new_width] = resized

# Convert BGR to RGB and normalize
image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
image_normalized = image_rgb.astype(np.float32) / 255.0

# Convert to NCHW format
input_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input tensor dtype: {input_tensor.dtype}")

# Run inference with all outputs
outputs = session.run(output_names, {input_name: input_tensor})
print(f"Number of outputs: {len(outputs)}")

for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}")
    print(f"Output {i} data type: {output.dtype}")
    print(f"Output {i} min/max values: {output.min():.4f} / {output.max():.4f}")
    
    # Show some sample values from the output
    if output.ndim >= 3:
        print(f"Output {i} sample values from first channel: {output[0, 0, :5, :5] if output.ndim == 4 else output[0, :5, :5]}")
    print()

# Save outputs for further analysis if needed
print("Inference completed successfully!")
print(f"Scale factor used: {scale:.3f} (for mapping outputs back to original image coordinates)")

# python draft/test_table_res_1.py 2> /dev/null