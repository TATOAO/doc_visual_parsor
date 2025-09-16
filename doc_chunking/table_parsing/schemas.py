"""
Pydantic schemas for table parsing and structure recognition.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import html


class TableType(str, Enum):
    """Types of tables based on structure."""
    WIRELESS = "wireless"  # Tables without visible grid lines
    WIRED = "wired"        # Tables with visible grid lines
    MIXED = "mixed"        # Tables with both wired and wireless sections
    UNKNOWN = "unknown"


class CellType(str, Enum):
    """Types of table cells."""
    HEADER = "header"
    DATA = "data"
    MERGED = "merged"
    EMPTY = "empty"


class TableAlignment(str, Enum):
    """Table alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class TableCell(BaseModel):
    """Represents a single table cell."""
    row: int = Field(..., description="Row index (0-based)", ge=0)
    col: int = Field(..., description="Column index (0-based)", ge=0)
    text: str = Field(default="", description="Text content of the cell")
    cell_type: CellType = Field(default=CellType.DATA, description="Type of cell")
    bbox: Optional[Tuple[float, float, float, float]] = Field(None, description="Bounding box (x1, y1, x2, y2)")
    confidence: Optional[float] = Field(None, description="Confidence score", ge=0, le=1)
    rowspan: int = Field(default=1, description="Number of rows this cell spans", ge=1)
    colspan: int = Field(default=1, description="Number of columns this cell spans", ge=1)
    is_merged: bool = Field(default=False, description="Whether this cell is merged")
    style: Optional[Dict[str, Any]] = Field(None, description="Cell styling information")
    
    @field_validator('text')
    @classmethod
    def escape_html_text(cls, v):
        """Escape HTML characters in text content."""
        if v:
            return html.escape(str(v))
        return v
    
    @property
    def is_empty(self) -> bool:
        """Check if cell is empty."""
        return not self.text or not self.text.strip()
    
    @property
    def is_header(self) -> bool:
        """Check if cell is a header cell."""
        return self.cell_type == CellType.HEADER


class TableRow(BaseModel):
    """Represents a table row."""
    index: int = Field(..., description="Row index (0-based)", ge=0)
    cells: List[TableCell] = Field(default_factory=list, description="Cells in this row")
    height: Optional[float] = Field(None, description="Row height in pixels")
    is_header: bool = Field(default=False, description="Whether this is a header row")
    
    @property
    def cell_count(self) -> int:
        """Number of cells in this row."""
        return len(self.cells)
    
    def get_cell(self, col_index: int) -> Optional[TableCell]:
        """Get cell at specific column index."""
        for cell in self.cells:
            if cell.col == col_index:
                return cell
        return None


class TableColumn(BaseModel):
    """Represents a table column."""
    index: int = Field(..., description="Column index (0-based)", ge=0)
    width: Optional[float] = Field(None, description="Column width in pixels")
    alignment: TableAlignment = Field(default=TableAlignment.LEFT, description="Column alignment")
    is_header: bool = Field(default=False, description="Whether this is a header column")
    
    def get_cells(self, rows: List[TableRow]) -> List[TableCell]:
        """Get all cells in this column."""
        cells = []
        for row in rows:
            cell = row.get_cell(self.index)
            if cell:
                cells.append(cell)
        return cells


class TableStructure(BaseModel):
    """Represents the structure of a table."""
    rows: List[TableRow] = Field(default_factory=list, description="Table rows")
    columns: List[TableColumn] = Field(default_factory=list, description="Table columns")
    row_count: int = Field(default=0, description="Number of rows", ge=0)
    col_count: int = Field(default=0, description="Number of columns", ge=0)
    has_header: bool = Field(default=False, description="Whether table has header row(s)")
    table_type: TableType = Field(default=TableType.UNKNOWN, description="Type of table")
    
    @field_validator('rows', mode='after')
    @classmethod
    def update_row_count(cls, v):
        """Update row count when rows are set."""
        if v:
            return v
        return v
    
    @field_validator('columns', mode='after')
    @classmethod
    def update_col_count(cls, v):
        """Update column count when columns are set."""
        if v:
            return v
        return v
    
    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Get cell at specific row and column."""
        if 0 <= row < self.row_count and 0 <= col < self.col_count:
            table_row = self.rows[row] if row < len(self.rows) else None
            if table_row:
                return table_row.get_cell(col)
        return None
    
    def get_row(self, row_index: int) -> Optional[TableRow]:
        """Get row at specific index."""
        if 0 <= row_index < len(self.rows):
            return self.rows[row_index]
        return None
    
    def get_column(self, col_index: int) -> Optional[TableColumn]:
        """Get column at specific index."""
        if 0 <= col_index < len(self.columns):
            return self.columns[col_index]
        return None
    
    def to_html(self, include_style: bool = True) -> str:
        """Convert table structure to HTML."""
        if not self.rows:
            return ""
        
        html_parts = ["<table>"]
        
        if include_style:
            html_parts.append('<style>')
            html_parts.append('table { border-collapse: collapse; width: 100%; }')
            html_parts.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
            html_parts.append('th { background-color: #f2f2f2; font-weight: bold; }')
            html_parts.append('</style>')
        
        for row in self.rows:
            html_parts.append("  <tr>")
            for cell in row.cells:
                tag = "th" if cell.is_header else "td"
                rowspan_attr = f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else ""
                colspan_attr = f' colspan="{cell.colspan}"' if cell.colspan > 1 else ""
                html_parts.append(f"    <{tag}{rowspan_attr}{colspan_attr}>{cell.text}</{tag}>")
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        return "\n".join(html_parts)
    
    def to_markdown(self) -> str:
        """Convert table structure to Markdown."""
        if not self.rows:
            return ""
        
        markdown_parts = []
        
        for i, row in enumerate(self.rows):
            # Create row content
            row_cells = []
            for cell in row.cells:
                # Handle merged cells by repeating content
                for _ in range(cell.colspan):
                    row_cells.append(cell.text or "")
            
            # Join cells with pipe separator
            markdown_parts.append("| " + " | ".join(row_cells) + " |")
            
            # Add header separator after first row if it's a header
            if i == 0 and self.has_header:
                separator = "| " + " | ".join(["---"] * len(row_cells)) + " |"
                markdown_parts.append(separator)
        
        return "\n".join(markdown_parts)
    
    def to_csv(self, delimiter: str = ",") -> str:
        """Convert table structure to CSV format."""
        if not self.rows:
            return ""
        
        csv_parts = []
        
        for row in self.rows:
            row_cells = []
            for cell in row.cells:
                # Handle merged cells by repeating content
                for _ in range(cell.colspan):
                    # Escape CSV special characters
                    text = cell.text or ""
                    if delimiter in text or '"' in text or '\n' in text:
                        text = f'"{text.replace('"', '""')}"'
                    row_cells.append(text)
            
            csv_parts.append(delimiter.join(row_cells))
        
        return "\n".join(csv_parts)


class TableElement(BaseModel):
    """Represents a detected table element with structure and content."""
    id: int = Field(..., description="Unique identifier")
    bbox: Tuple[float, float, float, float] = Field(..., description="Bounding box (x1, y1, x2, y2)")
    confidence: float = Field(..., description="Detection confidence", ge=0, le=1)
    structure: Optional[TableStructure] = Field(None, description="Table structure")
    html_content: Optional[str] = Field(None, description="HTML representation")
    markdown_content: Optional[str] = Field(None, description="Markdown representation")
    csv_content: Optional[str] = Field(None, description="CSV representation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @property
    def width(self) -> float:
        """Table width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Table height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Table area."""
        return self.width * self.height
    
    @property
    def has_structure(self) -> bool:
        """Check if table has structure information."""
        return self.structure is not None and self.structure.rows
    
    def get_text_content(self, separator: str = " ") -> str:
        """Get all text content from the table."""
        if not self.structure:
            return ""
        
        text_parts = []
        for row in self.structure.rows:
            row_text = []
            for cell in row.cells:
                if cell.text:
                    row_text.append(cell.text)
            if row_text:
                text_parts.append(separator.join(row_text))
        
        return "\n".join(text_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class TableParsingResult(BaseModel):
    """Container for table parsing results."""
    tables: List[TableElement] = Field(default_factory=list, description="Detected tables")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @property
    def table_count(self) -> int:
        """Number of detected tables."""
        return len(self.tables)
    
    @property
    def has_tables(self) -> bool:
        """Check if any tables were detected."""
        return len(self.tables) > 0
    
    def get_tables_by_type(self, table_type: TableType) -> List[TableElement]:
        """Get tables of specific type."""
        return [table for table in self.tables 
                if table.structure and table.structure.table_type == table_type]
    
    def get_largest_table(self) -> Optional[TableElement]:
        """Get the largest table by area."""
        if not self.tables:
            return None
        return max(self.tables, key=lambda t: t.area)
    
    def get_tables_with_structure(self) -> List[TableElement]:
        """Get tables that have structure information."""
        return [table for table in self.tables if table.has_structure]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


# Export all models
__all__ = [
    "TableType",
    "CellType", 
    "TableAlignment",
    "TableCell",
    "TableRow",
    "TableColumn",
    "TableStructure",
    "TableElement",
    "TableParsingResult"
]
