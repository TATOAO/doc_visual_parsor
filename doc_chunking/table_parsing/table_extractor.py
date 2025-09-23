"""
Table extraction combining PaddleOCR cell detection with PDF wire drawing and pdfplumber extraction.

This module implements a comprehensive table extraction solution that:
1. Uses PaddleOCR to detect table cells in images (for tables without visible grid lines)
2. Draws "wires" (grid lines) on the PDF based on detected cell positions
3. Uses pdfplumber to extract structured table data from the wire-enhanced PDF
4. Combines the old table processing logic for text and style extraction
"""

import os
import cv2
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import fitz  # PyMuPDF
import pdfplumber
from doc_chunking.schemas import LayoutElement
from PIL import Image
import io

# Import PaddleOCR for cell detection
try:
    from paddleocr import TableCellsDetection
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Table cell detection will be disabled.")

# Import project schemas
from .schemas import TableElement, TableStructure, TableCell, TableRow, TableColumn, TableType, CellType, TableParsingResult
from ..schemas import BoundingBox, StyleInfo, FontInfo, RunInfo

logger = logging.getLogger(__name__)


class PaddleOCRCellDetector:
    """
    PaddleOCR-based cell detector for identifying table cells in wireless tables.
    """
    
    def __init__(self, 
                 model_name: str = "RT-DETR-L_wireless_table_cell_det",
                 threshold: float = 0.3,
                 batch_size: int = 1):
        """
        Initialize PaddleOCR cell detector.
        
        Args:
            model_name: PaddleOCR model name for cell detection
            threshold: Detection confidence threshold
            batch_size: Batch size for inference
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is required for cell detection. Please install it with: pip install paddleocr")
        
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.model = None
        
        # Initialize model lazily
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the PaddleOCR model."""
        try:
            logger.info(f"Initializing PaddleOCR model: {self.model_name}")
            self.model = TableCellsDetection(model_name=self.model_name)
            logger.info("PaddleOCR model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR model: {str(e)}")
            raise

    def extract_table_image(self, pdf_path: Union[str, Path], layout_element: LayoutElement) -> str:
        """
        Extract table image from PDF using the same coordinate system as merging.py

        
        Args:
            pdf_path: Path to PDF file
            layout_element: Layout element containing table bbox and metadata
            
        Returns:
            Path to temporary image file
        """
        page_number = layout_element.metadata["page_number"]
        bbox = layout_element.bbox
        
        # Open PDF and get the specific page
        # maybe save some time if we pass a opened pdf document
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        
        # Convert bbox coordinates back to PDF coordinate system (72 DPI)
        # The bbox coordinates are in 150 DPI, so we need to scale them back to 72 DPI
        scale_factor = 150.0 / 72.0  # Same as in merging.py
        pdf_bbox = fitz.Rect(
            bbox.x1 / scale_factor,
            bbox.y1 / scale_factor, 
            bbox.x2 / scale_factor,
            bbox.y2 / scale_factor
        )
        
        # Calculate zoom factor for 150 DPI (same as merging.py)
        zoom = 150.0 / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render the page with the specific bbox clipped
        pix = page.get_pixmap(matrix=mat, clip=pdf_bbox)
        
        # Save to temporary file
        temp_fd, temp_image_path = tempfile.mkstemp(suffix='.png', prefix='table_page_')
        os.close(temp_fd)
        pix.save(temp_image_path)
        
        # Close the document
        doc.close()
        
        return temp_image_path

    
    def detect_cells(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect table cells in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected cells with bounding boxes and confidence scores
        """
        if self.model is None:
            raise RuntimeError("PaddleOCR model not initialized")
        
        try:
            # Run cell detection
            logger.info(f"Running cell detection on: {image_path}")
            output = self.model.predict(str(image_path), threshold=self.threshold, batch_size=self.batch_size)

            # for debug
            # output[0].save_to_img("./output/")
            
            # Extract cell information
            cells = []
            for res in output:
                # Check if result is a dictionary with 'boxes' key
                if isinstance(res, dict) and 'boxes' in res:
                    boxes = res['boxes']
                    for i, box_info in enumerate(boxes):
                        # Extract coordinates and score from the box info
                        if 'coordinate' in box_info and 'score' in box_info:
                            coords = box_info['coordinate']
                            score = box_info['score']
                            
                            # Convert box format (x1, y1, x2, y2)
                            x1, y1, x2, y2 = coords[:4] if len(coords) >= 4 else (0, 0, 0, 0)
                            
                            cell_info = {
                                'id': i,
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'confidence': float(score),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            cells.append(cell_info)
                
                # Fallback: Check if result has attributes (old format)
                elif hasattr(res, 'boxes') and hasattr(res, 'scores'):
                    for i, (box, score) in enumerate(zip(res.boxes, res.scores)):
                        # Convert box format (assuming x1, y1, x2, y2)
                        x1, y1, x2, y2 = box[:4] if len(box) >= 4 else (0, 0, 0, 0)
                        
                        cell_info = {
                            'id': i,
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'confidence': float(score),
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        cells.append(cell_info)
                
                # Alternative: if the result has different structure
                elif hasattr(res, 'print'):
                    # This might be the case for some PaddleOCR versions
                    # We'll need to extract from the result object differently
                    logger.warning("Alternative result format detected, extracting cells differently")
                    # Add logic here if needed based on actual PaddleOCR output format
            
            logger.info(f"Detected {len(cells)} cells")
            return cells
            
        except Exception as e:
            logger.error(f"Cell detection failed: {str(e)}")
            raise


class PDFWireDrawer:
    """
    Draws grid lines (wires) on PDF documents based on detected cell positions.
    """
    
    def __init__(self, line_width: float = 0.5, line_color: Tuple[float, float, float] = (0, 0, 0)):
        """
        Initialize PDF wire drawer.
        
        Args:
            line_width: Width of drawn lines in points
            line_color: RGB color of lines (0-1 range)
        """
        self.line_width = line_width
        self.line_color = line_color
    
    def _merge_close_coordinates(self, coords: List[float], tolerance: float = 2.0) -> List[float]:
        """
        Merge coordinates that are close to each other to avoid duplicate wires.
        
        Args:
            coords: Sorted list of coordinates
            tolerance: Distance threshold for merging (in points)
            
        Returns:
            List of merged coordinates
        """
        if not coords:
            return []
        
        merged = [coords[0]]
        
        for coord in coords[1:]:
            if abs(coord - merged[-1]) > tolerance:
                merged.append(coord)
            # If coordinates are close, we keep the previous one (no change to merged)
        
        return merged
    
    def draw_wires_on_pdf(self, 
                         pdf_path: Union[str, Path], 
                         cells: List[Dict[str, Any]], 
                         page_num: int = 0,
                         table_bbox: Optional[Tuple[float, float, float, float]] = None,
                         output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Draw grid lines on a PDF based on detected cells.
        
        Args:
            pdf_path: Path to the input PDF
            cells: List of detected cells with bounding boxes
            page_num: Page number to process (0-indexed)
            table_bbox: Optional bounding box to constrain wire drawing (x1, y1, x2, y2)
            output_path: Output path for the modified PDF (optional)
            
        Returns:
            Path to the modified PDF
        """
        if not cells:
            logger.warning("No cells provided for wire drawing")
            return str(pdf_path)
        
        # Open the PDF
        doc = fitz.open(str(pdf_path))
        
        if page_num >= len(doc):
            logger.error(f"Page {page_num} does not exist in PDF (total pages: {len(doc)})")
            doc.close()
            return str(pdf_path)
        
        page = doc[page_num]
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        logger.info(f"Page dimensions: {page_width} x {page_height}")
        
        # Extract unique x and y coordinates from cells
        x_coords = set()
        y_coords = set()
        
        for cell in cells:
            bbox = cell['bbox']
            x1, y1, x2, y2 = bbox
            x_coords.update([x1, x2])
            y_coords.update([y1, y2])
        
        # Merge close coordinates to avoid duplicate wires
        x_coords = self._merge_close_coordinates(sorted(x_coords))
        y_coords = self._merge_close_coordinates(sorted(y_coords))
        
        logger.info(f"After merging: {len(x_coords)} vertical lines and {len(y_coords)} horizontal lines")
        
        # Determine drawing bounds
        if table_bbox:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = table_bbox
            # Ensure coordinates are within the table region
            x_coords = [x for x in x_coords if bbox_x1 <= x <= bbox_x2]
            y_coords = [y for y in y_coords if bbox_y1 <= y <= bbox_y2]
            
            # Use table bounds for line drawing
            min_x, max_x = bbox_x1, bbox_x2
            min_y, max_y = bbox_y1, bbox_y2
        else:
            # Use page bounds
            min_x, max_x = 0, page_width
            min_y, max_y = 0, page_height
        
        logger.info(f"Drawing region: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        # Draw vertical lines (constrained to table region)
        for x in x_coords:
            if min_x <= x <= max_x:
                start_point = fitz.Point(x, min_y)
                end_point = fitz.Point(x, max_y)
                page.draw_line(start_point, end_point, color=self.line_color, width=self.line_width)
        
        # Draw horizontal lines (constrained to table region)
        for y in y_coords:
            if min_y <= y <= max_y:
                start_point = fitz.Point(min_x, y)
                end_point = fitz.Point(max_x, y)
                page.draw_line(start_point, end_point, color=self.line_color, width=self.line_width)
        
        # Save the modified PDF
        if output_path is None:
            # Create temporary file
            temp_fd, output_path = tempfile.mkstemp(suffix='.pdf', prefix='wired_table_')
            os.close(temp_fd)
        
        doc.save(str(output_path))
        doc.close()
        
        logger.info(f"Wired PDF saved to: {output_path}")
        return str(output_path)


class TableExtractor:
    """
    Main table extractor that combines PaddleOCR cell detection, PDF wire drawing, and pdfplumber extraction.
    """
    
    def __init__(self, 
                 cell_detector: Optional[PaddleOCRCellDetector] = None,
                 wire_drawer: Optional[PDFWireDrawer] = None,
                 confidence_threshold: float = 0.3):
        """
        Initialize table extractor.
        
        Args:
            cell_detector: PaddleOCR cell detector instance
            wire_drawer: PDF wire drawer instance
            confidence_threshold: Minimum confidence for cell detection
        """
        self.cell_detector = cell_detector or PaddleOCRCellDetector(threshold=confidence_threshold)
        self.wire_drawer = wire_drawer or PDFWireDrawer()
        self.confidence_threshold = confidence_threshold
    
    def _pdf_to_image(self, pdf_path: Union[str, Path], page_num: int = 0, dpi: int = 150) -> str:
        """
        Convert PDF page to image for cell detection.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to convert (0-indexed)
            dpi: DPI for image conversion
            
        Returns:
            Path to the generated image
        """
        doc = fitz.open(str(pdf_path))
        
        if page_num >= len(doc):
            raise ValueError(f"Page {page_num} does not exist in PDF (total pages: {len(doc)})")
        
        page = doc[page_num]
        
        # Convert page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale to desired DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Save to temporary file
        temp_fd, temp_image_path = tempfile.mkstemp(suffix='.png', prefix='table_page_')
        os.close(temp_fd)
        
        pix.save(temp_image_path)
        doc.close()
        
        logger.info(f"PDF page converted to image: {temp_image_path}")
        return temp_image_path
    
    def _char_in_bbox(self, char: dict, bbox: tuple) -> bool:
        """Check if a character is within a bounding box."""
        v_mid = (char["top"] + char["bottom"]) / 2
        h_mid = (char["x0"] + char["x1"]) / 2
        x0, top, x1, bottom = bbox
        return bool(
            (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
        )
    
    def _extract_cell_text_and_style(self, cell_bbox: tuple, chars: List[dict]) -> Tuple[str, StyleInfo]:
        """Extract text and style information from a cell (adapted from old processor)."""
        # Filter characters that are in this cell
        cell_chars = [char for char in chars if self._char_in_bbox(char, cell_bbox)]
        
        if not cell_chars:
            return "", StyleInfo(runs=[])
        
        # Sort characters by reading order (top to bottom, left to right)
        cell_chars.sort(key=lambda c: (c["top"], c["x0"]))
        
        # Extract text
        text = "".join(char["text"] for char in cell_chars)
        
        # Extract style information
        runs = []
        current_run = None
        
        for char in cell_chars:
            # Create font info for this character
            color_value = char.get("non_stroking_color", "#000000")
            if isinstance(color_value, tuple):
                # Convert RGB tuple to hex string
                r, g, b = [int(c * 255) for c in color_value]
                color_value = f"#{r:02x}{g:02x}{b:02x}"
            elif not isinstance(color_value, str):
                color_value = "#000000"
            
            font_info = FontInfo(
                name=char.get("fontname", "Unknown"),
                size=char.get("size", 0.0),
                bold=char.get("fontname", "").lower().find("bold") != -1,
                italic=char.get("fontname", "").lower().find("italic") != -1,
                underline=False,
                color=color_value
            )
            
            # Check if this character has the same style as the current run
            if (current_run is None or 
                current_run.font.name != font_info.name or
                current_run.font.size != font_info.size or
                current_run.font.bold != font_info.bold or
                current_run.font.italic != font_info.italic):
                
                # Start a new run
                if current_run is not None:
                    runs.append(current_run)
                
                current_run = RunInfo(
                    text=char["text"],
                    font=font_info,
                    start_index=len("".join(r.text for r in runs)),
                    end_index=len("".join(r.text for r in runs)) + len(char["text"])
                )
            else:
                # Extend current run
                current_run.text += char["text"]
                current_run.end_index = len("".join(r.text for r in runs)) + len(current_run.text)
        
        # Add the last run
        if current_run is not None:
            runs.append(current_run)
        
        # Create style info
        style_info = StyleInfo(
            runs=runs,
            primary_font=runs[0].font if runs else FontInfo(name="Unknown", size=0.0)
        )
        
        return text, style_info
    
    def extract_table_from_layout(self, 
                                  pdf_path: Union[str, Path], 
                                  layout_element: LayoutElement) -> Optional[TableStructure]:
        """
        Complete table extraction workflow: detect cells → draw wires → extract with pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            layout_element: Layout element containing table bbox and metadata
            
        Returns:
            TableStructure object or None if extraction failed
        """
        try:
            page_num = layout_element.metadata["page_number"]
            logger.info(f"Starting table extraction for page {page_num}")
            
            # Step 1: Extract table image from PDF
            temp_image_path = self.cell_detector.extract_table_image(pdf_path, layout_element)
            logger.info(f"Table image extracted: {temp_image_path}")
            
            # Step 2: Detect cells in the image
            cells = self.cell_detector.detect_cells(temp_image_path)
            if not cells:
                logger.warning("No cells detected in the table image")
                os.unlink(temp_image_path)  # Clean up
                return None
            
            logger.info(f"Detected {len(cells)} cells")
            
            # Step 3: Convert cell coordinates from image space to PDF space
            pdf_cells = self._convert_cells_to_pdf_coords(cells, layout_element)
            
            # Step 4: Draw wires on the PDF (constrained to table region)
            # Convert layout element bbox to tuple for wire drawing bounds
            table_bbox_tuple = (
                layout_element.bbox.x1 / (150.0 / 72.0),  # Convert to 72 DPI
                layout_element.bbox.y1 / (150.0 / 72.0),
                layout_element.bbox.x2 / (150.0 / 72.0),
                layout_element.bbox.y2 / (150.0 / 72.0)
            )
            
            wired_pdf_path = self.wire_drawer.draw_wires_on_pdf(
                pdf_path, pdf_cells, page_num, table_bbox_tuple
            )
            logger.info(f"Wires drawn on PDF: {wired_pdf_path}")
            
            # Debug: Save debug info
            debug_info = {
                'original_cells': cells,
                'pdf_cells': pdf_cells,
                'table_bbox': layout_element.bbox.model_dump(),
                'page_num': page_num
            }
            with open(f"debug_cells_{page_num}.json", "w") as f:
                json.dump(debug_info, f, indent=2, default=str)
            logger.info(f"Debug info saved to debug_cells_{page_num}.json")
            
            # Step 5: Extract table structure using pdfplumber
            table_structure = self._pdfplumber_extract_table(wired_pdf_path, page_num)
            
            # Clean up temporary files
            os.unlink(temp_image_path)
            # Don't delete the wired PDF for debugging
            # if wired_pdf_path != str(pdf_path):  # Only delete if it's a temp file
            #     os.unlink(wired_pdf_path)
            
            return table_structure
            
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
            return None
    
    def _convert_cells_to_pdf_coords(self, 
                                   cells: List[Dict[str, Any]], 
                                   layout_element: LayoutElement) -> List[Dict[str, Any]]:
        """
        Convert cell coordinates from image space to PDF coordinate space.
        
        Args:
            cells: List of detected cells with image coordinates
            layout_element: Layout element containing table bbox and metadata
            
        Returns:
            List of cells with PDF coordinates
        """
        bbox = layout_element.bbox
        
        # The cell coordinates are in image space (150 DPI)
        # We need to convert them back to PDF space and offset by table position
        
        # Scale factor from 150 DPI back to 72 DPI
        scale_factor = 72.0 / 150.0
        
        # Table position in PDF coordinates (already in 72 DPI from merging.py conversion)
        table_x_offset = bbox.x1 / (150.0 / 72.0)  # Convert back to 72 DPI
        table_y_offset = bbox.y1 / (150.0 / 72.0)  # Convert back to 72 DPI
        
        pdf_cells = []
        for cell in cells:
            image_bbox = cell['bbox']
            x1, y1, x2, y2 = image_bbox
            
            # Convert to PDF coordinates and offset by table position
            pdf_x1 = x1 * scale_factor + table_x_offset
            pdf_y1 = y1 * scale_factor + table_y_offset
            pdf_x2 = x2 * scale_factor + table_x_offset
            pdf_y2 = y2 * scale_factor + table_y_offset
            
            pdf_cell = cell.copy()
            pdf_cell['bbox'] = (pdf_x1, pdf_y1, pdf_x2, pdf_y2)
            pdf_cells.append(pdf_cell)
        
        logger.info(f"Converted {len(pdf_cells)} cells to PDF coordinates")
        return pdf_cells

    def _pdfplumber_extract_table(self, pdf_path: Union[str, Path], page_num: int = 0) -> Optional[TableStructure]:
        """
        Extract table structure using pdfplumber from a wire-enhanced PDF.
        
        Args:
            pdf_path: Path to the wire-enhanced PDF
            page_num: Page number to process (0-indexed)
            
        Returns:
            TableStructure object or None if no table found
        """
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_num >= len(pdf.pages):
                    logger.error(f"Page {page_num} does not exist in PDF (total pages: {len(pdf.pages)})")
                    return None
                
                page = pdf.pages[page_num]
                
                # Find tables on the page
                tables = page.find_tables()
                
                if not tables:
                    logger.warning("No tables found with pdfplumber")
                    return None
                
                # Use the first (largest) table
                table = tables[0]
                chars = page.chars
                
                logger.info(f"Found table with {len(table.rows)} rows")
                logger.info(f"Table bbox: {table.bbox}")
                logger.info(f"Found {len(chars)} characters on page")
                
                # Process table rows and cells
                table_rows = []
                all_cells = []
                
                for row_idx, row in enumerate(table.rows):
                    row_cells = []
                    
                    for col_idx, cell_bbox in enumerate(row.cells):
                        if cell_bbox is None:
                            # Empty cell
                            cell = TableCell(
                                row=row_idx,
                                col=col_idx,
                                text="",
                                cell_type=CellType.EMPTY,
                                bbox=(0, 0, 1, 1),
                                confidence=1.0
                            )
                        else:
                            # Extract text and style for this cell
                            cell_text, cell_style = self._extract_cell_text_and_style(cell_bbox, chars)
                            
                            # Determine cell type (simple heuristic)
                            cell_type = CellType.HEADER if row_idx == 0 else CellType.DATA
                            
                            cell = TableCell(
                                row=row_idx,
                                col=col_idx,
                                text=cell_text.strip(),
                                cell_type=cell_type,
                                bbox=cell_bbox,
                                confidence=1.0,  # pdfplumber doesn't provide confidence
                                style={'font_info': cell_style.model_dump() if cell_style else None}
                            )
                        
                        row_cells.append(cell)
                        all_cells.append(cell)
                    
                    table_row = TableRow(
                        index=row_idx,
                        cells=row_cells,
                        is_header=(row_idx == 0)
                    )
                    table_rows.append(table_row)
                
                # Create table structure
                table_structure = TableStructure(
                    rows=table_rows,
                    columns=[TableColumn(index=i) for i in range(len(table_rows[0].cells) if table_rows else 0)],
                    row_count=len(table_rows),
                    col_count=len(table_rows[0].cells) if table_rows else 0,
                    has_header=True if table_rows else False,
                    table_type=TableType.WIRED  # Now it has wires drawn
                )
                
                logger.info(f"Extracted table structure: {table_structure.row_count}x{table_structure.col_count}")
                return table_structure
                
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {str(e)}")
            return None




def test_table_extractor():
    """Test the complete table extraction workflow."""
    table_extractor = TableExtractor()
    pdf_path = "3900.pdf"
    
    # Load the layout element from the result file
    element_data = json.load(open("result_2.json", "r", encoding="utf-8"))[0]
    element = LayoutElement(**element_data)
    
    print(f"Testing table extraction for page {element.metadata['page_number']}")
    print(f"Table bbox: {element.bbox}")
    
    # Use the complete workflow
    table_structure = table_extractor.extract_table_from_layout(pdf_path, element)
    
    if table_structure:
        print(f"Successfully extracted table: {table_structure.row_count}x{table_structure.col_count}")
        print(f"Table type: {table_structure.table_type}")
        print(f"Has header: {table_structure.has_header}")
        
        # Print first few rows for verification
        for i, row in enumerate(table_structure.rows):
            print(f"Row {i}: {[cell.text for cell in row.cells]}")
    else:
        print("Table extraction failed")


def main():
    table_extractor = PaddleOCRCellDetector()
    table_extractor._initialize_model()
    pdf_path = "3900.pdf"
    element = json.load(open("result_2.json", "r", encoding="utf-8"))[0]
    element = LayoutElement(**element)
    # print(element)
    temp_image_path = table_extractor.extract_table_image(pdf_path, element)
    print(f"Table image saved to: {temp_image_path}")
    cells = table_extractor.detect_cells(temp_image_path)

    # draw boxes on elements
    print(f"Detected {len(cells)} cells")
    print(cells)

# python -m doc_chunking.table_parsing.table_extractor
if __name__ == "__main__":
    # main()
    test_table_extractor()