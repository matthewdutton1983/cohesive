# Import standard libraries
import io
import unittest

# Import third-party libraries
import networkx as nx
import numpy as np

# Import project code
from cohesive import CohesiveTextSegmenter


class TestCohesiveTextSegmenter(unittest.TestCase):
    def setUp(self):
        """Initialize a CohesiveTestSegmenter instance for testing."""
        self.segmenter = CohesiveTextSegmenter()


    def test_load_model_success(self):
        """Test if the model can be loaded successfully."""
        self.assertIsNotNone(self.segmenter.model)


    def test_generate_embeddings(self):
        """Test the _generate_embeddings method."""
        sentences = ["This is a test sentence.", "Another sentence for testing."]
        embeddings = self.segmenter._generate_embeddings(sentences)
        self.assertEqual(len(embeddings), len(sentences))
        self.assertEqual(embeddings.shape[1], 768)


    def test_normalize_similarities(self):
        """Test the _normalize_similarities method."""
        similarity_matrix = np.array([[0.5, 0.3], [0.8, 0.6]])
        embeddings = np.array([[1, 2], [3, 4]])
        normalized_matrix = self.segmenter._normalize_similarities(similarity_matrix, embeddings)
        expected_matrix = np.array([[0.18257419, 0.10954451], [0.18257419, 0.13608276]])
        np.testing.assert_almost_equal(normalized_matrix, expected_matrix)


    def test_create_similarity_matrix(self):
        """Test the _create_similarity_matrix method."""
        embeddings = np.array([[1, 2], [3, 4]])
        similarity_matrix = self.segmenter._create_similarity_matrix(embeddings)
        expected_matrix = np.array([[5, 11], [11, 25]])
        np.testing.assert_equal(similarity_matrix, expected_matrix)


    def test_calculate_local_similarities(self):
        """Test the _calculate_local_similarities method."""
        sentences = ["This is a test sentence.", "Another sentence for testing."]
        similarity_matrix = np.array([[0.5, 0.3], [0.8, 0.6]])
        local_similarities = self.segmenter._calculate_local_similarities(sentences, similarity_matrix)
        expected_matrix = np.array([[0.5, 0.3], [0.8, 0.6]])
        np.testing.assert_equal(local_similarities, expected_matrix)


    def test_calculate_combined_similarities(self):
        """Test the _calculate_combined_similarities method."""
        sentences = ["This is a test sentence.", "Another sentence for testing."]
        similarity_matrix = np.array([[0.5, 0.3], [0.8, 0.6]])
        local_similarities = np.array([[0.5, 0.3], [0.8, 0.6]])
        combined_similarities = self.segmenter._calculate_combined_similarities(sentences, similarity_matrix, local_similarities)
        expected_matrix = np.array([[0.65, 0.36], [0.88, 0.64]])
        np.testing.assert_equal(combined_similarities, expected_matrix)


    def test_create_nx_graph(self):
        """Test the _create_nx_graph method."""
        similarity_matrix = np.array([[0.5, 0.3], [0.8, 0.6]])
        nx_graph = self.segmenter._create_nx_graph(similarity_matrix)
        self.assertIsInstance(nx_graph, nx.Graph)
        self.assertEqual(nx_graph.number_of_nodes(), similarity_matrix.shape[0])


    def test_find_best_partition(self):
        """Test the _find_best_partition method."""
        nx_graph = nx.Graph([(0, 1, {'weight': 0.5}), (1, 2, {'weight': 0.8}), (2, 0, {'weight': 0.3})])
        partition = self.segmenter._find_best_partition(nx_graph)
        self.assertIsInstance(partition, dict)
        self.assertEqual(set(partition.values()), {0, 1, 2})


    def test_merge_clusters(self):
        """Test the _merge_clusters method."""
        clusters = [np.array([1, 2]), np.array([2, 3]), np.array([4, 5])]
        merged_clusters = self.segmenter._merge_clusters(clusters)
        expected_clusters = [np.array([1, 2, 3]), np.array([4, 5])]
        
        for merged, expected in zip(merged_clusters, expected_clusters):
            np.testing.assert_equal(merged, expected)


    def test_group_nodes_into_segments(self):
        """Test the _group_nodes_into_segments method."""
        partition = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        segments = self.segmenter._group_nodes_into_segments(partition)
        expected_segments = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
        
        for segment, expected in zip(segments, expected_segments):
            np.testing.assert_equal(segment, expected)


    def test_extract_segment_boundaries(self):
        """Test the _extract_segment_boundaries method."""
        segment = [(0, "This is a test."), (1, "Another test.")]
        start, end = self.segmenter._extract_segment_boundaries(segment)
        self.assertEqual(start, 0)
        self.assertEqual(end, 1)


    def test_clean_text(self):
        """Test the _clean_text method."""
        dirty_text = "This\tis\na\n\n   test! "
        cleaned_text = self.segmenter._clean_text(dirty_text)
        self.assertEqual(cleaned_text, "This is a test!")


    def test_update_parameters(self):
        """Test the _update_parameters method."""
        self.segmenter._update_parameters(alpha=0.7, context_window=8, decay=0.9, resolution=1.2)
        self.assertEqual(self.segmenter.alpha, 0.7)
        self.assertEqual(self.segmenter.context_window, 8)
        self.assertEqual(self.segmenter.decay, 0.9)
        self.assertEqual(self.segmenter.resolution, 1.2)


    def test_create_segments_empty_sentences(self):
        """Test create_segments with empty sentences."""
        with self.assertRaises(ValueError):
            self.segmenter.create_segments([], alpha=0.7)


    def test_create_segments_valid_input(self):
        """Test create_segments with valid input."""
        sentences = ["This is a test sentence.", "Another sentence for testing."]
        result = self.segmenter.create_segments(sentences, alpha=0.7)
        self.assertEqual(result, "Generated 2 segments.")
        self.assertIsNotNone(self.segmenter.segments)


    def test_create_segments_mismatched_lengths(self):
        """Test create_segments with mismatched lengths of sentences and embeddings."""
        sentences = ["This is a test sentence.", "Another sentence for testing."]
        with self.assertRaises(ValueError):
            self.segmenter.create_segments(sentences, alpha=0.7)


    def test_print_segments(self):
        """Test the print_segments method."""
        self.segmenter.segments = [(0, [(0, "This is a test."), (1, "Another test.")]),
                                    (1, [(2, "Yet another test.")])]
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
          self.segmenter.print_segments()
          output = mock_stdout.getvalue().strip()
          self.assertIn("This is a test. Another test.", output)
          self.assertIn("Yet another test.", output)


    def test_print_segment_boundaries(self):
        """Test the print_segment_boundaries method."""
        self.segmenter.segments = [(0, [(0, "This is a test."), (1, "Another test.")]),
                                    (1, [(2, "Yet another test.")])]
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            self.segmenter.print_segment_boundaries()
            output = mock_stdout.getvalue().strip()
            self.assertIn("(0, 1)", output)
            self.assertIn("(2, 2)", output)


    def test_get_params(self):
        """Test the get_params method."""
        params = self.segmenter.get_params()
        expected_params = {
            "alpha": self.segmenter.alpha,
            "context_window": self.segmenter.context_window,
            "decay": self.segmenter.decay,
            "resolution": self.segmenter.resolution
        }
        self.assertDictEqual(params, expected_params)


    def test_finetune_params(self):
        """Test the finetune_params method."""
        self.segmenter.finetune_params(alpha=0.7, context_window=8, decay=0.9, resolution=1.2)
        self.assertEqual(self.segmenter.alpha, 0.7)
        self.assertEqual(self.segmenter.context_window, 8)
        self.assertEqual(self.segmenter.decay, 0.9)
        self.assertEqual(self.segmenter.resolution, 1.2)


if __name__ == '__main__':
    unittest.main()
