import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

def preprocess_image(image, input_size=1024):
    """Preprocess image for SlanetPlus model."""
    original_height, original_width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image with padding value 114
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_height, :new_width] = resized
    
    # Convert BGR to RGB and normalize
    image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Convert to NCHW format
    input_tensor = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    
    return input_tensor, scale

def decode_predictions(outputs, scale, confidence_threshold=0.5):
    """
    Decode SlanetPlus outputs to get table structure elements.
    
    Args:
        outputs: List of model outputs
        scale: Scale factor used in preprocessing
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detected elements with their bounding boxes and types
    """
    # Output 0: (1, N, 8) - likely [x1, y1, x2, y2, conf, ...] format
    # Output 1: (1, N, 50) - classification scores for different element types
    
    boxes_output = outputs[0][0]  # Shape: (501, 8)
    class_output = outputs[1][0]  # Shape: (501, 50)
    
    detections = []
    
    for i in range(len(boxes_output)):
        # Extract box coordinates and confidence
        box_data = boxes_output[i]
        class_scores = class_output[i]
        
        # Assuming the format is [x1, y1, x2, y2, confidence, ...]
        # The exact format might need adjustment based on the model
        if len(box_data) >= 5:
            x1, y1, x2, y2 = box_data[:4]
            confidence = box_data[4]  # or might be max of class_scores
        else:
            # Alternative: confidence might be the max class score
            confidence = np.max(class_scores)
            # Box coordinates might be in a different format
            x1, y1, x2, y2 = box_data[:4]
        
        # Filter by confidence
        if confidence > confidence_threshold:
            # Scale coordinates back to original image size
            x1_orig = x1 / scale
            y1_orig = y1 / scale
            x2_orig = x2 / scale
            y2_orig = y2 / scale
            
            # Get predicted class (element type)
            predicted_class = np.argmax(class_scores)
            class_confidence = class_scores[predicted_class]
            
            detections.append({
                'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                'confidence': confidence,
                'class_id': predicted_class,
                'class_confidence': class_confidence,
                'type': get_element_type(predicted_class)
            })
    
    return detections

def get_element_type(class_id):
    """Map class ID to element type name."""
    # This mapping is based on common table structure elements
    # The exact mapping would depend on the model's training data
    element_types = {
        0: 'table',
        1: 'table_row', 
        2: 'table_column',
        3: 'table_cell',
        4: 'table_header',
        5: 'table_body',
        6: 'text_line',
        7: 'text_block'
    }
    return element_types.get(class_id, f'class_{class_id}')

def visualize_results(image_path, detections, output_path='table_structure_visualization.png'):
    """Visualize the detected table structure elements."""
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_rgb)
    
    # Define colors for different element types
    colors = {
        'table': 'red',
        'table_row': 'blue', 
        'table_column': 'green',
        'table_cell': 'yellow',
        'table_header': 'purple',
        'table_body': 'orange',
        'text_line': 'cyan',
        'text_block': 'magenta'
    }
    
    # Draw bounding boxes
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        width = x2 - x1
        height = y2 - y1
        
        element_type = detection['type']
        color = colors.get(element_type, 'gray')
        
        # Create rectangle
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = f"{element_type} ({detection['confidence']:.2f})"
        ax.text(x1, y1-5, label, fontsize=8, color=color, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title(f'Table Structure Detection Results ({len(detections)} elements found)')
    ax.axis('off')
    
    # Add legend
    legend_elements = [patches.Patch(color=color, label=element_type) 
                      for element_type, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")

def main():
    # Model and image paths
    model_path = "/Users/tatoao_mini/Work/doc_visual_parsor/model_parameters/layout_detection/OpenDataLab/PDF-Extract-Kit-1___0/models/TabRec/SlanetPlus/slanet-plus.onnx"
    image_path = "table_demo1_withline.png"
    
    # Load model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    input_tensor, scale = preprocess_image(image)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Scale factor: {scale:.3f}")
    
    # Run inference
    outputs = session.run(output_names, {input_name: input_tensor})
    print(f"Model outputs: {len(outputs)} tensors")
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")
    
    # Decode predictions with different confidence thresholds
    for conf_thresh in [0.1, 0.3, 0.5]:
        print(f"\n--- Results with confidence threshold {conf_thresh} ---")
        detections = decode_predictions(outputs, scale, confidence_threshold=conf_thresh)
        print(f"Found {len(detections)} detections")
        
        if detections:
            # Print detection details
            for i, det in enumerate(detections[:10]):  # Show first 10
                print(f"Detection {i}: {det['type']} at {det['bbox']} (conf: {det['confidence']:.3f})")
            
            # Visualize results
            output_file = f'table_structure_conf_{conf_thresh}.png'
            visualize_results(image_path, detections, output_file)
        else:
            print("No detections found at this confidence threshold")

if __name__ == "__main__":
    main()
