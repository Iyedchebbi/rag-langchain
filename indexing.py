from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from file_reader import read_multiple_pdfs
from openrouter_embeddings import OpenRouterEmbeddings


if __name__ == "__main__":
    embeddings = OpenRouterEmbeddings()
    vector_store = InMemoryVectorStore(embeddings)

    doc_paths = [
        "pdfs\\Iyed_Chebbi_Profile_and_Background.pdf",
        "pdfs\\Iyed_Chebbi_Skills_Projects_and_Business.pdf",
    ]

    docs = read_multiple_pdfs(doc_paths)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print(f"Split documents into {len(all_splits)} chunks ")

    vector_store.add_documents(all_splits)
    print(f"All {len(all_splits)} chunks are Indexed")
