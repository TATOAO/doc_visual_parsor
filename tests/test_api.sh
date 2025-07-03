# chunk docx
curl -X POST http://localhost:8000/api/chunk-document -F "file=@/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# chunk docx sse 
curl -X POST http://localhost:8000/api/chunk-document-sse -F "file=@/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# chunk pdf see
curl -X POST http://localhost:8000/api/chunk-document-sse -F "file=@/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.pdf;type=application/pdf"