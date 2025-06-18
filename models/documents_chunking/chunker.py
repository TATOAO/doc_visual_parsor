from models.schemas.schemas import

class Chunker:
    def __init__(self):
        pass

    def chunk(self, document: Document):
        pass

if __name__ == "__main__":
    chunker = Chunker()
    chunker.chunk(Document(
        file_path="test.pdf",
        file_type="pdf"
    ))