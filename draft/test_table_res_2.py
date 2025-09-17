import onnxruntime as ort
import cv2
import numpy as np

model_path = "/Users/tatoao_mini/Work/doc_visual_parsor/model_parameters/layout_detection/OpenDataLab/PDF-Extract-Kit-1___0/models/TabRec/UnetStructure/unet.onnx"
image_path = "table_demo1_withline.png"
# image_path = "demo_table_wo_line.png"

session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image")
    exit(1)

print(f"Original image shape: {image.shape}")

# Preprocessing for TableMaster table structure recognition
# Model expects dynamic dimensions but typically works with 480x480
input_size = 480  # Use 480 as default size for TableMaster

original_height, original_width = image.shape[:2]
print(f"Original image size: {original_width}x{original_height}")

# Resize image while maintaining aspect ratio
scale = min(input_size / original_width, input_size / original_height)
new_width = int(original_width * scale)
new_height = int(original_height * scale)

print(f"Scale factor: {scale:.3f}")
print(f"Resized dimensions: {new_width}x{new_height}")

# Calculate padding offsets (important for coordinate mapping!)
pad_x = 0  # No horizontal padding since we place at top-left
pad_y = 0  # No vertical padding since we place at top-left

# Resize image
resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Create padded image with padding value 114
padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized

print(f"Padding offset: x={pad_x}, y={pad_y}")
print(f"Active region in 480x480: ({pad_x},{pad_y}) to ({pad_x+new_width},{pad_y+new_height})")

# Convert to float and normalize (typical preprocessing for ONNX models)
# Convert BGR to RGB and normalize to [0, 1] range
padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
padded_float = padded_rgb.astype(np.float32) / 255.0

# Add batch dimension and transpose to NCHW format (batch, channels, height, width)
input_tensor = np.transpose(padded_float, (2, 0, 1))  # HWC to CHW
input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input tensor dtype: {input_tensor.dtype}")
print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

result = session.run(output_names, {input_name: input_tensor})

# Post-process the segmentation result
def interpret_unet_structure_output(result, original_image_shape, scale, resized_width, resized_height, pad_x, pad_y):
    """
    Interpret UnetStructure model output (segmentation mask).
    
    Args:
        result: Raw ONNX model output
        original_image_shape: Shape of original input image
        scale: Scale factor used in preprocessing
        resized_width: Width after resizing (before padding)
        resized_height: Height after resizing (before padding)
        pad_x: Horizontal padding offset
        pad_y: Vertical padding offset
    
    Returns:
        Dictionary with structure analysis
    """
    if not result or len(result) == 0:
        return {"error": "No output from model"}
    
    # Get the segmentation mask
    segmentation_mask = result[0]  # Shape: (1, 1, H, W) or similar
    
    # Remove batch and channel dimensions if present
    while len(segmentation_mask.shape) > 2:
        segmentation_mask = segmentation_mask[0]
    
    print(f"Segmentation mask shape: {segmentation_mask.shape}")
    print(f"Unique values in mask: {np.unique(segmentation_mask)}")
    
    # Map pixel values to table structure elements
    # Based on common UNet table structure models:
    element_mapping = {
        0: "background",
        1: "table_cell",  # or table body
        2: "table_border",  # or table lines
        3: "table_header",  # if present
        4: "text_content"   # if present
    }
    
    # Analyze the segmentation mask
    analysis = {
        "mask_shape": segmentation_mask.shape,
        "unique_values": list(np.unique(segmentation_mask)),
        "element_counts": {},
        "element_percentages": {},
        "detected_elements": []
    }
    
    # Count pixels for each element type
    total_pixels = segmentation_mask.size
    for value in np.unique(segmentation_mask):
        count = np.sum(segmentation_mask == value)
        element_type = element_mapping.get(int(value), f"unknown_class_{int(value)}")
        analysis["element_counts"][element_type] = int(count)
        analysis["element_percentages"][element_type] = float(count / total_pixels * 100)
        
        if value > 0:  # Skip background
            analysis["detected_elements"].append(element_type)
    
    # Detect potential table structure
    if 1 in segmentation_mask:  # Table cells detected
        analysis["has_table_structure"] = True
        analysis["table_coverage"] = float(np.sum(segmentation_mask > 0) / total_pixels * 100)
    else:
        analysis["has_table_structure"] = False
        analysis["table_coverage"] = 0.0
    
    return analysis, segmentation_mask

# Interpret the results with proper coordinate mapping
analysis, mask = interpret_unet_structure_output(result, image.shape, scale, new_width, new_height, pad_x, pad_y)

print("\n=== TABLE STRUCTURE ANALYSIS ===")
print(f"Has table structure: {analysis['has_table_structure']}")
print(f"Table coverage: {analysis['table_coverage']:.1f}%")
print(f"Detected elements: {analysis['detected_elements']}")

print("\n=== ELEMENT BREAKDOWN ===")
for element, percentage in analysis["element_percentages"].items():
    if percentage > 1.0:  # Only show significant elements
        print(f"{element}: {percentage:.1f}%")

# Save the segmentation mask as an image for visualization
def save_segmentation_visualization(mask, original_image, resized_width, resized_height, pad_x, pad_y, output_path="table_structure_segmentation.png"):
    """Save a visualization of the segmentation mask."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    print(f"Creating visualization...")
    print(f"Mask shape: {mask.shape}, values: {np.unique(mask)}")
    print(f"Original image shape: {original_image.shape}")
    
    # Create a more distinct colormap
    unique_vals = np.unique(mask)
    colors_dict = {
        0: [0, 0, 0],        # Black for background
        1: [255, 0, 0],      # Bright red for table cells
        2: [0, 255, 0],      # Bright green for table borders
        3: [0, 0, 255],      # Blue for headers
        4: [255, 255, 0],    # Yellow for other elements
    }
    
    # Create visualization with 2x2 subplots for better layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original image
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Raw segmentation mask (enhanced contrast)
    # Create custom colormap for better visibility
    cmap_colors = ['black', 'red', 'lime', 'blue', 'yellow'][:len(unique_vals)]
    cmap = mcolors.ListedColormap(cmap_colors)
    
    im2 = ax2.imshow(mask, cmap=cmap, vmin=0, vmax=len(unique_vals)-1)
    ax2.set_title('Segmentation Mask (480x480)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add text annotations showing what each color means
    legend_text = []
    for i, val in enumerate(unique_vals):
        if val == 0:
            legend_text.append(f'{val}: Background')
        elif val == 1:
            legend_text.append(f'{val}: Table Cells')
        elif val == 2:
            legend_text.append(f'{val}: Table Borders')
        else:
            legend_text.append(f'{val}: Class {val}')
    
    ax2.text(0.02, 0.98, '\n'.join(legend_text), transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # 3. Overlay on original image with cell boundaries
    # Extract only the active region from the mask (remove padding)
    active_mask = mask[pad_y:pad_y+resized_height, pad_x:pad_x+resized_width]
    
    # Resize the active mask to match original image dimensions
    mask_resized = cv2.resize(active_mask.astype(np.uint8), 
                             (original_rgb.shape[1], original_rgb.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Create overlay with cell boundaries
    overlay_with_cells = original_rgb.copy()
    
    # First, add subtle colored regions
    for val in unique_vals:
        if val > 0:  # Skip background
            color = colors_dict.get(val, [255, 255, 255])
            mask_binary = (mask_resized == val)
            # Make overlay more transparent for better visibility
            overlay_with_cells[mask_binary] = (overlay_with_cells[mask_binary] * 0.8 + 
                                             np.array(color) * 0.2).astype(np.uint8)
    
    # Now detect and draw cell boundaries with blue lines
    # Find connected components in cell regions (value 1)
    cell_mask = (mask_resized == 1).astype(np.uint8)
    
    try:
        from scipy import ndimage
        labeled_cells, num_cells = ndimage.label(cell_mask)
        
        print(f"Drawing {num_cells} cell boundaries...")
        
        # Draw boundary for each cell
        for i in range(1, num_cells + 1):
            cell_region = (labeled_cells == i).astype(np.uint8)
            
            # Find contours of this cell
            contours, _ = cv2.findContours(cell_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours with blue lines
            for contour in contours:
                cv2.drawContours(overlay_with_cells, [contour], -1, (0, 0, 255), 2)  # Blue lines, thickness 2
                
                # Also draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay_with_cells, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Blue rectangle
                
    except ImportError:
        print("scipy not available, drawing simple grid instead")
        # Fallback: draw grid based on detected borders
        border_mask = (mask_resized == 2).astype(np.uint8)
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_with_cells, contours, -1, (0, 0, 255), 2)
    
    ax3.imshow(overlay_with_cells)
    ax3.set_title('Structure with Cell Boundaries (Blue Lines)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Enhanced mask analysis
    # Create a more detailed visualization showing individual regions
    enhanced_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val in unique_vals:
        color = colors_dict.get(val, [128, 128, 128])
        mask_binary = (mask == val)
        enhanced_mask[mask_binary] = color
    
    ax4.imshow(enhanced_mask)
    ax4.set_title('Enhanced Segmentation View', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Add statistics text
    stats_text = []
    total_pixels = mask.size
    for val in unique_vals:
        count = np.sum(mask == val)
        percentage = (count / total_pixels) * 100
        if val == 0:
            element = "Background"
        elif val == 1:
            element = "Cells"
        elif val == 2:
            element = "Borders"
        else:
            element = f"Class {val}"
        stats_text.append(f'{element}: {percentage:.1f}%')
    
    ax4.text(0.02, 0.98, '\n'.join(stats_text), transform=ax4.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Enhanced visualization saved to: {output_path}")
    plt.close()
    
    # Also save individual mask as a separate clear image
    mask_output = output_path.replace('.png', '_mask_only.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(enhanced_mask)
    plt.title('Table Structure Segmentation Mask', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Add legend
    legend_elements = []
    for val in unique_vals:
        color = [c/255.0 for c in colors_dict.get(val, [128, 128, 128])]
        if val == 0:
            label = "Background"
        elif val == 1:
            label = "Table Cells"
        elif val == 2:
            label = "Table Borders"
        else:
            label = f"Class {val}"
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.savefig(mask_output, dpi=150, bbox_inches='tight')
    print(f"Mask-only visualization saved to: {mask_output}")
    plt.close()

# Extract more detailed table structure
def extract_table_structure(mask, scale, resized_width, resized_height, pad_x, pad_y):
    """Extract detailed table structure from segmentation mask."""
    # Find table borders (value 2) to identify grid structure
    borders = (mask == 2).astype(np.uint8)
    cells = (mask == 1).astype(np.uint8)
    
    structure_info = {
        "grid_analysis": {},
        "cell_regions": [],
        "estimated_rows": 0,
        "estimated_cols": 0
    }
    
    # Analyze horizontal and vertical lines
    # Sum along axes to find line patterns
    horizontal_projection = np.sum(borders, axis=1)  # Sum along width
    vertical_projection = np.sum(borders, axis=0)    # Sum along height
    
    # Find peaks in projections (indicating table lines)
    def find_lines(projection, min_length=10):
        lines = []
        for i, value in enumerate(projection):
            if value > min_length:  # Threshold for line detection
                lines.append(i)
        return lines
    
    horizontal_lines = find_lines(horizontal_projection)
    vertical_lines = find_lines(vertical_projection)
    
    # Estimate rows and columns from line spacing
    if len(horizontal_lines) > 1:
        # Group nearby lines together
        h_groups = []
        current_group = [horizontal_lines[0]]
        for line in horizontal_lines[1:]:
            if line - current_group[-1] < 5:  # Lines within 5 pixels are same line
                current_group.append(line)
            else:
                h_groups.append(current_group)
                current_group = [line]
        h_groups.append(current_group)
        structure_info["estimated_rows"] = max(1, len(h_groups) - 1)
    
    if len(vertical_lines) > 1:
        # Group nearby lines together
        v_groups = []
        current_group = [vertical_lines[0]]
        for line in vertical_lines[1:]:
            if line - current_group[-1] < 5:  # Lines within 5 pixels are same line
                current_group.append(line)
            else:
                v_groups.append(current_group)
                current_group = [line]
        v_groups.append(current_group)
        structure_info["estimated_cols"] = max(1, len(v_groups) - 1)
    
    # Find connected components in cell regions
    from scipy import ndimage
    try:
        labeled_cells, num_cells = ndimage.label(cells)
        structure_info["detected_cell_regions"] = int(num_cells)
        
        # Get properties of each cell region
        cell_regions = []
        for i in range(1, num_cells + 1):
            cell_mask = (labeled_cells == i)
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) > 0:
                # Adjust coordinates to account for padding and scaling
                # First remove padding offset
                x_coords_adjusted = x_coords - pad_x
                y_coords_adjusted = y_coords - pad_y
                
                # Filter out coordinates that fall outside the active region
                valid_indices = (x_coords_adjusted >= 0) & (x_coords_adjusted < resized_width) & \
                               (y_coords_adjusted >= 0) & (y_coords_adjusted < resized_height)
                
                if np.sum(valid_indices) > 0:
                    x_coords_valid = x_coords_adjusted[valid_indices]
                    y_coords_valid = y_coords_adjusted[valid_indices]
                    
                    # Scale back to original image coordinates
                    cell_bbox = {
                        "x1": int(np.min(x_coords_valid) / scale),
                        "y1": int(np.min(y_coords_valid) / scale),
                        "x2": int(np.max(x_coords_valid) / scale),
                        "y2": int(np.max(y_coords_valid) / scale),
                        "area": int(np.sum(valid_indices))
                    }
                    cell_regions.append(cell_bbox)
        
        structure_info["cell_regions"] = cell_regions
        
    except ImportError:
        print("Note: scipy not available, skipping connected components analysis")
        structure_info["detected_cell_regions"] = "N/A (scipy required)"
    
    structure_info["grid_analysis"] = {
        "horizontal_lines_detected": len(horizontal_lines),
        "vertical_lines_detected": len(vertical_lines),
        "has_grid_structure": len(horizontal_lines) > 1 and len(vertical_lines) > 1
    }
    
    return structure_info

# Extract detailed structure
structure_info = extract_table_structure(mask, scale, new_width, new_height, pad_x, pad_y)

print("\n=== DETAILED STRUCTURE ANALYSIS ===")
print(f"Estimated table dimensions: {structure_info['estimated_rows']} rows × {structure_info['estimated_cols']} columns")
print(f"Detected cell regions: {structure_info['detected_cell_regions']}")
print(f"Has grid structure: {structure_info['grid_analysis']['has_grid_structure']}")
print(f"Horizontal lines: {structure_info['grid_analysis']['horizontal_lines_detected']}")
print(f"Vertical lines: {structure_info['grid_analysis']['vertical_lines_detected']}")

if structure_info.get("cell_regions") and len(structure_info["cell_regions"]) > 0:
    print(f"\n=== CELL REGIONS (scaled to original image) ===")
    for i, cell in enumerate(structure_info["cell_regions"][:5]):  # Show first 5 cells
        print(f"Cell {i+1}: x={cell['x1']}-{cell['x2']}, y={cell['y1']}-{cell['y2']}, area={cell['area']} pixels")
    if len(structure_info["cell_regions"]) > 5:
        print(f"... and {len(structure_info['cell_regions']) - 5} more cells")

# Create visualization
save_segmentation_visualization(mask, image, new_width, new_height, pad_x, pad_y)

print(f"\n=== SUMMARY ===")
print(f"✅ Table detected: {analysis['has_table_structure']}")
print(f"📊 Structure coverage: {analysis['table_coverage']:.1f}%")
print(f"🏗️  Grid structure: {structure_info['grid_analysis']['has_grid_structure']}")
print(f"📏 Estimated size: {structure_info['estimated_rows']}×{structure_info['estimated_cols']}")
print(f"🔍 Cell regions found: {structure_info['detected_cell_regions']}")
print(f"🎨 Visualization: table_structure_segmentation.png")

# Extract cell bounding boxes for text mapping
def extract_cell_positions(mask, resized_width, resized_height, pad_x, pad_y, scale):
    """
    Extract cell bounding boxes that can be used to map existing PDF text to cells.
    
    Returns:
        List of cell bounding boxes in original image coordinates
    """
    print("\n=== EXTRACTING CELL POSITIONS ===")
    
    # Extract only the active region from the mask
    active_mask = mask[pad_y:pad_y+resized_height, pad_x:pad_x+resized_width]
    
    cell_positions = []
    
    try:
        from scipy import ndimage
        
        # Find cell regions (value 1 in the mask)
        cells = (active_mask == 1).astype(np.uint8)
        labeled_cells, num_cells = ndimage.label(cells)
        
        print(f"Found {num_cells} cell regions")
        
        for i in range(1, num_cells + 1):
            cell_region = (labeled_cells == i)
            y_coords, x_coords = np.where(cell_region)
            
            if len(y_coords) > 0 and len(x_coords) > 0:
                # Get bounding box in resized coordinates
                x1_resized = int(np.min(x_coords))
                y1_resized = int(np.min(y_coords))
                x2_resized = int(np.max(x_coords))
                y2_resized = int(np.max(y_coords))
                
                # Scale back to original image coordinates
                x1_orig = int(x1_resized / scale)
                y1_orig = int(y1_resized / scale)
                x2_orig = int(x2_resized / scale)
                y2_orig = int(y2_resized / scale)
                
                # Skip very small regions (likely noise)
                width = x2_orig - x1_orig
                height = y2_orig - y1_orig
                if width < 10 or height < 10:
                    continue
                
                cell_bbox = {
                    'cell_id': len(cell_positions) + 1,
                    'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),  # (x1, y1, x2, y2)
                    'width': width,
                    'height': height,
                    'area': width * height,
                    'center': ((x1_orig + x2_orig) // 2, (y1_orig + y2_orig) // 2)
                }
                
                cell_positions.append(cell_bbox)
        
        # Sort cells by position (top-to-bottom, left-to-right for table structure)
        cell_positions.sort(key=lambda c: (c['center'][1] // 50, c['center'][0]))
        
        print(f"Extracted {len(cell_positions)} valid cell positions")
        
        return cell_positions
        
    except ImportError:
        print("scipy not available for cell position extraction")
        return []

# Extract cell positions
cell_positions = extract_cell_positions(mask, new_width, new_height, pad_x, pad_y, scale)

# Display cell positions
if cell_positions:
    print(f"\n=== CELL POSITIONS FOR TEXT MAPPING ===")
    for cell in cell_positions:
        x1, y1, x2, y2 = cell['bbox']
        print(f"Cell {cell['cell_id']:2d}: bbox=({x1:4d},{y1:3d},{x2:4d},{y2:3d}) "
              f"size={cell['width']:3d}x{cell['height']:2d} center=({cell['center'][0]:4d},{cell['center'][1]:3d})")
    
    print(f"\n📊 Total cell positions extracted: {len(cell_positions)}")
    print(f"💡 You can now map PDF text elements to these cell regions based on coordinates")
    print(f"🔍 Use spatial overlap to assign text to the correct cells")

# Save cell positions to JSON for easy integration
import json

def save_cell_positions_json(cell_positions, output_file="cell_positions.json"):
    """Save cell positions to JSON file for easy integration."""
    if not cell_positions:
        return
    
    # Convert to serializable format
    json_data = {
        'total_cells': len(cell_positions),
        'cells': []
    }
    
    for cell in cell_positions:
        json_data['cells'].append({
            'cell_id': cell['cell_id'],
            'bbox': cell['bbox'],
            'width': cell['width'],
            'height': cell['height'],
            'center': cell['center'],
            'area': cell['area']
        })
    
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📄 Cell positions saved to: {output_file}")

# Save positions to JSON
save_cell_positions_json(cell_positions)