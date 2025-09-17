
import onnxruntime as ort
import cv2
import numpy as np

model_path = "/Users/tatoao_mini/Work/doc_visual_parsor/model_parameters/layout_detection/OpenDataLab/PDF-Extract-Kit-1___0/models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx"
image_path = "table_demo1_withline.png"
# image_path = "/Users/tatoao_mini/Work/doc_visual_parsor/demo_table_wo_line.png"
session = ort.InferenceSession(model_path)

# Get input and output information
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name
output_name = output_info.name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")
print(f"Input shape: {input_info.shape}")
print(f"Input type: {input_info.type}")
print(f"Output shape: {output_info.shape}")
print(f"Output type: {output_info.type}")

print(f"Available providers: {session.get_providers()}")

# Load and preprocess image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image")
    exit(1)

print(f"Original image shape: {image.shape}")

# Preprocessing based on model requirements
# Model expects: [batch_size, 3, 224, 224] in CHW format

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to expected input size (224x224)
image_resized = cv2.resize(image_rgb, (224, 224))

# Convert to float32 and normalize to [0, 1]
image_normalized = image_resized.astype(np.float32) / 255.0

# Convert from HWC to CHW format (Height, Width, Channels -> Channels, Height, Width)
image_chw = np.transpose(image_normalized, (2, 0, 1))

# Add batch dimension: [C, H, W] -> [1, C, H, W]
input_tensor = np.expand_dims(image_chw, axis=0)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input tensor dtype: {input_tensor.dtype}")

# Run inference - note the correct format: list of output names, not single string
result = session.run([output_name], {input_name: input_tensor})
print(f"Result: {result}")
print(f"Result shape: {result[0].shape if result else 'No result'}")

# Interpret results (this appears to be a binary classification model)
if result and len(result) > 0:
    predictions = result[0][0]  # Get first batch result
    print(f"Predictions: {predictions}")
    
    # Assuming this is a table classification model (table vs non-table)
    if len(predictions) == 2:
        prob_not_table = predictions[0]
        prob_is_table = predictions[1]
        
        print(f"Probability NOT table: {prob_not_table:.4f}")
        print(f"Probability IS table: {prob_is_table:.4f}")
        
        predicted_class = "TABLE" if prob_is_table > prob_not_table else "NOT_TABLE"
        confidence = max(prob_is_table, prob_not_table)
        
        print(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")

# python draft/test.py