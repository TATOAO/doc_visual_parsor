# given a "Section Tree Object" (which extracted from the raw text) find the position of the section title in the raw text

from doc_chunking.schemas.schemas import Section, Positions
from typing import List, Tuple

from fuzzysearch import find_near_matches

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
