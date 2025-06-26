import re
from typing import List, Optional, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from doc_chunking.schemas.layout_schemas import LayoutElement




class TitleNumberType(str, Enum):
    """Types of title numbering systems"""
    CHINESE_UNIT = "chinese_unit"  # 第N条, 第N款, etc.
    CHINESE_PARENTHETICAL = "chinese_parenthetical"  # （一）, （二）, etc.
    ARABIC_DECIMAL = "arabic_decimal"  # 1., 2., 17.1, etc.
    ARABIC_PARENTHETICAL = "arabic_parenthetical"  # (1), (2), etc.
    MIXED_ALPHANUMERIC = "mixed_alphanumeric"  # 1.1.a, a.i, etc.
    ROMAN_NUMERAL = "roman_numeral"  # I, II, III, etc.
    ALPHABETIC = "alphabetic"  # a, b, c, etc.


class TitleNumberUnit(str, Enum):
    条 = "条"
    款 = "款"
    项 = "项"
    目 = "目"
    章 = "章"
    节 = "节"
    部 = "部"
    编 = "编"
    篇 = "篇"
    number = "number"


class TitleNumber(BaseModel):
    """Represents a parsed title number with hierarchical information"""
    raw_text: str = Field(description="Original text containing the title number")
    number_type: TitleNumberType = Field(description="Type of numbering system")
    hierarchy_parts: List[Union[int, str]] = Field(description="Hierarchical parts of the number")
    unit: Optional[TitleNumberUnit] = Field(default=None, description="Unit if applicable")
    prefix: Optional[str] = Field(default=None, description="Prefix like '第'")
    suffix: Optional[str] = Field(default=None, description="Suffix text")
    level: int = Field(description="Hierarchical level (0-based)")
    
    def __str__(self) -> str:
        return f"{self.number_type}:{'.'.join(map(str, self.hierarchy_parts))}"


def chinese_number_to_arabic(chinese_num: str) -> int:
    """Convert Chinese numbers to Arabic numbers"""
    chinese_digits = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000, '万': 10000
    }
    
    if chinese_num in chinese_digits:
        return chinese_digits[chinese_num]
    
    # Handle complex Chinese numbers like 六百一十
    result = 0
    temp = 0
    
    for char in chinese_num:
        if char in chinese_digits:
            val = chinese_digits[char]
            if val >= 10:
                if val == 10 and temp == 0:
                    temp = 1
                temp *= val
                if val >= 10000:
                    result += temp
                    temp = 0
            else:
                temp += val
        
    return result + temp


def roman_to_arabic(roman: str) -> int:
    """Convert Roman numerals to Arabic numbers"""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0
    
    for char in reversed(roman.upper()):
        value = roman_values.get(char, 0)
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    
    return result


def title_number_extraction(text: str) -> Optional[TitleNumber]:
    """
    Extract the title number from the text using various patterns.
    
    Handles patterns like:
    - 第六百一十条 (Chinese with unit)
    - （一） (Chinese parenthetical)  
    - 17.1 (Arabic decimal)
    - (1) (Arabic parenthetical)
    - 1.1.a (Mixed alphanumeric)
    - a.i (Alphabetic with sub-numbering)
    - I, II, III (Roman numerals)
    """
    if not text or not text.strip():
        return None
        
    text = text.strip()
    
    # Pattern 1: Chinese with unit (第N条, 第N款, etc.)
    chinese_unit_pattern = r'^第([零一二三四五六七八九十百千万]+)([条款项目章节部编篇])'
    match = re.match(chinese_unit_pattern, text)
    if match:
        chinese_num, unit = match.groups()
        arabic_num = chinese_number_to_arabic(chinese_num)
        return TitleNumber(
            raw_text=text,
            number_type=TitleNumberType.CHINESE_UNIT,
            hierarchy_parts=[arabic_num],
            unit=TitleNumberUnit(unit),
            prefix="第",
            level=0
        )
    
    # Pattern 2: Chinese parenthetical （一）, （二）, etc.
    chinese_paren_pattern = r'^[（(]([一二三四五六七八九十百千万零]+)[）)]'
    match = re.match(chinese_paren_pattern, text)
    if match:
        chinese_num = match.group(1)
        arabic_num = chinese_number_to_arabic(chinese_num)
        return TitleNumber(
            raw_text=text,
            number_type=TitleNumberType.CHINESE_PARENTHETICAL,
            hierarchy_parts=[arabic_num],
            level=1
        )
    
    # Pattern 3: Arabic parenthetical (1), (2), etc.
    arabic_paren_pattern = r'^[（(](\d+)[）)]'
    match = re.match(arabic_paren_pattern, text)
    if match:
        num = int(match.group(1))
        return TitleNumber(
            raw_text=text,
            number_type=TitleNumberType.ARABIC_PARENTHETICAL,
            hierarchy_parts=[num],
            level=1
        )
    
    # Pattern 4: Complex decimal/alphanumeric (1.1.a, a.i, 17.1, etc.)
    complex_pattern = r'^([a-zA-Z]*\d*(?:\.[a-zA-Z\d]+)*(?:\.[a-zA-Z]+(?:\.[ivxlcdm]+)?)?)'
    match = re.match(complex_pattern, text)
    if match:
        number_part = match.group(1)
        parts = number_part.split('.')
        hierarchy_parts = []
        
        for part in parts:
            if part.isdigit():
                hierarchy_parts.append(int(part))
            elif part.isalpha():
                if part.lower() in ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']:
                    hierarchy_parts.append(roman_to_arabic(part))
                else:
                    hierarchy_parts.append(part.lower())
            else:
                # Mixed alphanumeric
                hierarchy_parts.append(part)
        
        # Determine the type based on content
        if any(isinstance(part, str) and not part.isdigit() for part in hierarchy_parts):
            number_type = TitleNumberType.MIXED_ALPHANUMERIC
        else:
            number_type = TitleNumberType.ARABIC_DECIMAL
            
        return TitleNumber(
            raw_text=text,
            number_type=number_type,
            hierarchy_parts=hierarchy_parts,
            level=len(hierarchy_parts) - 1
        )
    
    # Pattern 5: Simple number at start (1. 2. etc.)
    simple_number_pattern = r'^(\d+)\.?\s'
    match = re.match(simple_number_pattern, text)
    if match:
        num = int(match.group(1))
        return TitleNumber(
            raw_text=text,
            number_type=TitleNumberType.ARABIC_DECIMAL,
            hierarchy_parts=[num],
            level=0
        )
    
    # Pattern 6: Roman numerals
    roman_pattern = r'^([IVXLCDM]+)\.?\s'
    match = re.match(roman_pattern, text)
    if match:
        roman_num = match.group(1)
        arabic_num = roman_to_arabic(roman_num)
        return TitleNumber(
            raw_text=text,
            number_type=TitleNumberType.ROMAN_NUMERAL,
            hierarchy_parts=[arabic_num],
            level=0
        )
    
    # Pattern 7: Single alphabetic (a., b., etc.)
    alpha_pattern = r'^([a-zA-Z])\.?\s'
    match = re.match(alpha_pattern, text)
    if match:
        letter = match.group(1).lower()
        return TitleNumber(
            raw_text=text,
            number_type=TitleNumberType.ALPHABETIC,
            hierarchy_parts=[letter],
            level=0
        )
    
    return None


def judge_title_number_sibling(title_number1: TitleNumber, title_number2: TitleNumber) -> bool:
    """
    Judge if two title numbers are siblings (same hierarchical level and parent).
    """
    if not title_number1 or not title_number2:
        return False
    
    # Must have same level
    if title_number1.level != title_number2.level:
        return False
    
    # Must have same number of hierarchy parts
    if len(title_number1.hierarchy_parts) != len(title_number2.hierarchy_parts):
        return False
    
    # For single-level items, just check if they're the same type
    if len(title_number1.hierarchy_parts) == 1:
        return (title_number1.number_type == title_number2.number_type and
                title_number1.unit == title_number2.unit)
    
    # For multi-level items, all parent parts must be identical
    parent_parts1 = title_number1.hierarchy_parts[:-1]
    parent_parts2 = title_number2.hierarchy_parts[:-1]
    
    return parent_parts1 == parent_parts2


def judge_title_number_parent_child(parent: TitleNumber, child: TitleNumber) -> bool:
    """
    Judge if one title number is the direct parent of another.
    """
    if not parent or not child:
        return False
    
    # Child must be one level deeper
    if child.level != parent.level + 1:
        return False
    
    # Child must have one more hierarchy part
    if len(child.hierarchy_parts) != len(parent.hierarchy_parts) + 1:
        return False
    
    # All parent parts must match the beginning of child parts
    return child.hierarchy_parts[:-1] == parent.hierarchy_parts


def get_title_hierarchy_path(title_number: TitleNumber) -> str:
    """
    Get a string representation of the hierarchical path.
    """
    if not title_number:
        return ""
    
    path_parts = []
    for part in title_number.hierarchy_parts:
        if isinstance(part, int):
            path_parts.append(str(part))
        else:
            path_parts.append(str(part))
    
    return ".".join(path_parts)


def compare_title_numbers(title1: TitleNumber, title2: TitleNumber) -> str:
    """
    Compare two title numbers and return their relationship.
    Returns: 'sibling', 'parent_child', 'child_parent', 'unrelated'
    """
    if not title1 or not title2:
        return 'unrelated'
    
    if judge_title_number_sibling(title1, title2):
        return 'sibling'
    elif judge_title_number_parent_child(title1, title2):
        return 'parent_child'
    elif judge_title_number_parent_child(title2, title1):
        return 'child_parent'
    else:
        return 'unrelated'

