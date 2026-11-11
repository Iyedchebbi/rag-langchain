import uuid

from deepagents.backends import StateBackend
from langchain.tools import tool
from indexing import vector_store


# Initialize temporary agent filesystem
backend = StateBackend()


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """

    # 1. Search vector database
    retrieved_docs = vector_store.similarity_search(query, k=4)

    # 2. Generate unique ID for this search
    batch_id = uuid.uuid4().hex[:8]

    # 3. Prepare upload lists
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    # 4. Process retrieved documents
    for index, doc in enumerate(retrieved_docs, start=1):

        # Create Markdown file path
        path = f"/retrieved/{batch_id}/chunk_{index}.md"

        # Prepare file content
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )

        # Add path and content as a tuple
        uploads.append((path, content.encode("utf-8")))

        # Save path
        saved_paths.append(path)

    # 5. Upload files to agent filesystem
    if uploads:
        backend.upload_files(uploads)

    # 6. Return saved file paths
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )
