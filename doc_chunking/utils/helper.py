# given a "Section Tree Object" (which extracted from the raw text) find the position of the section title in the raw text

from doc_chunking.schemas import Section
from typing import List, Tuple

from rapidfuzz.distance import Levenshtein as RFLevenshtein
from typing import NamedTuple


class _NearMatch(NamedTuple):
    start: int
    end: int
    dist: int


def find_near_matches(pattern: str, text: str, max_l_dist: int = 3):
    """
    Minimal replacement for fuzzysearch.find_near_matches using RapidFuzz distances.
    Finds substring windows whose Levenshtein distance to pattern is <= max_l_dist.
    Returns a list of objects with .start and .end to match prior usage.
    """
    if not pattern or not text:
        return []

    pattern_len = len(pattern)
    text_len = len(text)
    min_len = max(1, pattern_len - max_l_dist)
    max_len = min(text_len, pattern_len + max_l_dist)

    candidates = []
    for window_len in range(min_len, max_len + 1):
        if window_len > text_len:
            break
        # Slide a window over the text and compute distance
        for i in range(0, text_len - window_len + 1):
            candidate = text[i : i + window_len]
            dist = RFLevenshtein.distance(pattern, candidate)
            if dist <= max_l_dist:
                candidates.append(_NearMatch(i, i + window_len, dist))

    # Sort by start position to mimic fuzzysearch ordering (closest in order)
    candidates.sort(key=lambda m: (m.start, m.dist))
    return candidates

def find_ordered_fuzzy_sequence(text: str, keywords: List[str], max_l_dist: int = 3) -> List[Tuple[int, int]]:
    """
    Search for keywords in order with fuzzy matching. Return list of (start, end) index for each match.

    ## TODO: This is a naive implementation, it failes when the keywords are not unique in the raw text
    """
    matches = []
    cursor = 0  # Where to start search in the text

    for keyword in keywords:
        # Search for keyword in remaining text with fuzzy match
        sub_text = text[cursor:]
        result = find_near_matches(keyword, sub_text, max_l_dist=max_l_dist)
        if not result:
            return []  # One keyword not found in order

        # Choose first match (closest in order)
        match = result[0]
        start, end = match.start + cursor, match.end + cursor
        matches.append((start, end))
        cursor = end  # Move cursor forward

    return matches


def remove_circular_references(section: Section):
    """
    Recursively remove parent_section references to avoid circular reference in JSON serialization
    """
    section.parent_section = None
    for sub_section in section.sub_sections:
        remove_circular_references(sub_section)


def detect_file_type(file_input):
    """
    Detect if the file is a PDF, DOCX, or DOC based on magic number.
    Accepts file path (str or Path) or bytes.
    Returns 'pdf', 'docx', 'doc', or None.
    """
    from pathlib import Path
    import os

    def _get_first_bytes(f, n=8):
        if isinstance(f, (str, Path)):
            with open(f, 'rb') as file:
                return file.read(n)
        elif isinstance(f, bytes):
            return f[:n]
        else:
            raise ValueError("Unsupported input type for detect_file_type")

    first_bytes = _get_first_bytes(file_input)
    # PDF: %PDF-
    if first_bytes.startswith(b'%PDF-'):
        return 'pdf'
    # DOCX: PK\x03\x04 (zip file)
    if first_bytes.startswith(b'PK\x03\x04'):
        return 'docx'
    # DOC (OLE Compound File): D0 CF 11 E0 A1 B1 1A E1
    if first_bytes.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):
        return 'doc'
    return None
