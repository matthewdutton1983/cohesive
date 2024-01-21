# Import standard libraries
import unittest
from unittest.mock import Mock, patch

# Import third-party libraries
import numpy as np
from sentence_transformers import SentenceTransformer

# Import project code
from cohesive.encoder import SentenceTransformersEncoder

class TestSentenceTransformersEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_model = Mock(spec=SentenceTransformer)
        cls.mock_model.encode.return_value = [np.array([1.0, 2.0, 3.0])]
        cls.model_name_or_path = "mock_model"


    def setUp(self):
        self.encoder = SentenceTransformersEncoder(model_name_or_path=self.model_name_or_path)


    def test_encoder_initialization_successful(self):
        self.assertEqual(self.encoder.model, self.mock_model)


    @patch.object(SentenceTransformer, '__init__', side_effect=Exception("Mocked exception"))
    def test_encoder_initialization_failure(self, mock_init):
        with self.assertRaises(ValueError) as context:
            SentenceTransformersEncoder(model_name_or_path="invalid_model")

        self.assertIn("Error loading model", str(context.exception))
        mock_init.assert_called_once_with("invalid_model")


    def test_generate_embeddings_successful(self):
        sentences = ["This is a test.", "Another test."]
        embeddings = self.encoder.generate_embeddings(sentences)
        self.assertEqual(len(embeddings), len(sentences))
        self.assertIsInstance(embeddings[0], np.ndarray)


    def test_generate_embeddings_empty_input(self):
        with self.assertRaises(ValueError) as context:
            self.encoder.generate_embeddings([])

        self.assertIn("Input sentences list is empty.", str(context.exception))


    def test_generate_embeddings_mismatched_lengths(self):
        sentences = ["This is a test.", "Another test."]
        with patch.object(SentenceTransformer, 'encode', return_value=[np.array([1.0, 2.0, 3.0])]):
            with self.assertRaises(ValueError) as context:
                self.encoder.generate_embeddings(sentences + ["Extra sentence"])
            self.assertIn("Number of sentences does not match number of embeddings.", str(context.exception))


    def test_reshape_embedding_1d(self):
        embedding = np.array([1.0, 2.0, 3.0])
        reshaped_embedding = self.encoder.reshape_embedding(embedding)
        self.assertEqual(reshaped_embedding.shape, (1, 3))


    def test_reshape_embedding_2d(self):
        embedding = np.array([[1.0, 2.0, 3.0]])
        reshaped_embedding = self.encoder.reshape_embedding(embedding)
        self.assertEqual(reshaped_embedding.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()
