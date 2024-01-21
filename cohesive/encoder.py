# Import standard libraries
from typing import List, Union, Optional

# Import third-party libraries
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformersEncoder:
    def __init__(self, model_name_or_path: str):
        """
        Initializes the SentenceTransformersEncoder class.

        Parameters:
        - model_name_or_path (str): The name or path of the Sentence Transformers model.
        """
        try:
            self.model: SentenceTransformer = SentenceTransformer(model_name_or_path)
        except Exception as e:
            raise ValueError("Error loading model: {}".format(e))


    def generate_embeddings(self, sentences: List[str], show_progress_bar: Optional[bool] = True) -> List[np.ndarray]:
        """
        Generates embeddings for a list of sentences using the loaded Sentence Transformers model.

        Parameters:
        - sentences (List[str]): List of sentences to generate embeddings for.
        - show_progress_bar (Optional[bool]): Flag to show a progress bar during encoding. Default is True.

        Returns:
        - List[np.ndarray]: List of embeddings corresponding to the input sentences.
        """
        if not sentences:
            raise ValueError("Input sentences list is empty.")

        embeddings: List[np.ndarray] = self.model.encode(sentences, show_progress_bar=show_progress_bar)

        if len(embeddings) != len(sentences):
            raise ValueError("Number of sentences does not match number of embeddings.")

        return embeddings


    def reshape_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Reshapes a given embedding.

        Parameters:
        - embedding (np.ndarray): The embedding to reshape.

        Returns:
        - np.ndarray: The reshaped embedding.
        """
        if len(embedding.shape) == 1:
            return embedding.reshape(1, -1)

        return embedding
  