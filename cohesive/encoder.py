# Import third-party libraries
from sentence_transformers import SentenceTransformer


class SentenceTransformersEncoder:
  def __init__(self, model_name_or_path):
    try:
      self.model = SentenceTransformer(model_name_or_path)
    except Exception as e:
      raise ValueError("Error loading model: {}".format(e))
    

  def generate_embeddings(self, sentences, show_progress_bar):
    if not sentences:
      raise ValueError("Input sentences list is empty.")
    
    embeddings = self.model.encode(sentences, show_progress_bar=show_progress_bar)

    if len(embeddings) != len(sentences):
      raise ValueError("Number of sentences does not match number of embeddings.")
    
    return embeddings
  
  
  def reshape_embedding(self, embedding):
    if len(embedding.shape) == 1:
      return embedding.reshape(1, -1)
    
    return embedding
  