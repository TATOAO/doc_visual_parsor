# bash tests/test_upload_file.sh

# curl -X POST http://localhost:8000/api/upload-document -F "file=@tests/test_data/1-1 买卖合同（通用版）.pdf"

# curl -X POST http://localhost:8000/api/analyze-pdf-structure -F "file=@tests/test_data/1-1 买卖合同（通用版）.pdf"

# curl -X POST http://localhost:8000/api/analyze-structure -F "file=@tests/test_data/1-1 买卖合同（通用版）.pdf"

# curl -X POST http://localhost:8000/api/extract-pdf-pages-into-images -F "file=@tests/test_data/1-1 买卖合同（通用版）.pdf"

# curl -X POST http://localhost:8000/api/extract-docx-content -F "file=@tests/test_data/1-1 买卖合同（通用版）.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document"


curl -X POST http://localhost:8000/api/analyze-docx-with-naive-llm -F "file=@tests/test_data/1-1 买卖合同（通用版）.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document" 