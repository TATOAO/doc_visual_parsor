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
            
            # Extract cell information
            cells = []
            for res in output:
                # Get detection results
                if hasattr(res, 'boxes') and hasattr(res, 'scores'):
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
    
    def draw_wires_on_pdf(self, 
                         pdf_path: Union[str, Path], 
                         cells: List[Dict[str, Any]], 
                         page_num: int = 0,
                         output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Draw grid lines on a PDF based on detected cells.
        
        Args:
            pdf_path: Path to the input PDF
            cells: List of detected cells with bounding boxes
            page_num: Page number to process (0-indexed)
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
        
        # Sort coordinates
        x_coords = sorted(x_coords)
        y_coords = sorted(y_coords)
        
        logger.info(f"Drawing {len(x_coords)} vertical lines and {len(y_coords)} horizontal lines")
        
        # Draw vertical lines
        for x in x_coords:
            if 0 <= x <= page_width:
                start_point = fitz.Point(x, 0)
                end_point = fitz.Point(x, page_height)
                page.draw_line(start_point, end_point, color=self.line_color, width=self.line_width)
        
        # Draw horizontal lines
        for y in y_coords:
            if 0 <= y <= page_height:
                start_point = fitz.Point(0, y)
                end_point = fitz.Point(page_width, y)
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





# python -m doc_chunking.table_parsing.table_extractor
if __name__ == "__main__":
    # pdf_layout_extractor = PdfLayoutExtractor()
    # result = pdf_layout_extractor.extract_layout("3800.pdf")

    from doc_chunking.merging import PdfStyleCVMixLayoutExtractor


    pdf_path = "/Users/tatoao_mini/Work/doc_visual_parsor/CFA ESG CURRICULUM 2024.pdf"
    async def main_page_by_page():
        pdf_style_cv_mix_layout_extractor = PdfStyleCVMixLayoutExtractor(
            model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx",
            cv_confidence_threshold=0.1  # Use lower threshold for better detection
        )
        i = 0
        async for result in pdf_style_cv_mix_layout_extractor.detect_layout_page_by_page(pdf_path):  # Test with first 3 pages
            print(f"Page {i}: {len(result)} elements")
            import json
            json.dump([r.model_dump() for r in result], open(f"result_{i}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
            i += 1

            if i == 3:
                break


    import asyncio
    asyncio.run(main_page_by_page())
    # main()