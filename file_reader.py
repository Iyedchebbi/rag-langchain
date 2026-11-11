from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader


def read_pdf_text(pdf_path: str | Path) -> str:
    '''read and return the text content of a pdf file'''
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF NOT FOUND {path}")
    
    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or ""  for page in reader.pages] 
    return "\n".join(text_parts)

def read_multiple_pdfs(pdf_paths : List[str | Path]) -> List [Document]:
    ''' read text from multiple pdfs and return a list of Documents '''
    docs : List [Document] = []
    for path in pdf_paths:
        path_obj = Path(path)
        text = read_pdf_text(path_obj)
        docs.append(Document(page_content=text, metadata = {"source" : str(path_obj)}))
    return docs


if __name__ == "__main__":
    doc_paths = [
        "pdfs\\Iyed_Chebbi_Profile_and_Background.pdf",
        "pdfs\\Iyed_Chebbi_Skills_Projects_and_Business.pdf",
    ]
    print(read_multiple_pdfs(doc_paths))
