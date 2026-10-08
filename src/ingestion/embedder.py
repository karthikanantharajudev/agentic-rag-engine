from typing import List
from google import genai
from config.settings import settings


class GeminiEmbedder:
    """Handles vector embedding generation using Google's Gemini API."""

    def __init__(self, model_name: str = settings.embedding_model):
        self.model_name = model_name
        self.client = genai.Client()

    def generate_embedding(self, text: str) -> List[float]:
        """Generates a 768-dimensional float vector for a single text string."""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return response.embeddings[0].values