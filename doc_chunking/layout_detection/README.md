
# Layout "Detection"
We assume user will only input pdf that has text (not a image version of pdf) or .docx.


# Method
using a CV to do layout detection and use pymupudf/ python-docx to enrich the text and style information back to the layout detected.


# Enhancement

1. Limitation for layout height; 某个box太长的问题