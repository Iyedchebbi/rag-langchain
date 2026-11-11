import os
import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class OpenRouterEmbeddings(Embeddings):

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "nvidia/nemotron-3-embed-1b:free"
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def _embed(self, texts):
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])

        return [item["embedding"] for item in data]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]
