# Import standard libraries
import unittest
from unittest.mock import Mock, patch

# Import third-party libraries
import community as community_louvain
import numpy as np
import networkx as nx

# Import project code
from cohesive import Cohesive
from cohesive.base import Segment, Sentence


class TestCohesive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_model = Mock()
        cls.mock_model.encode.return_value = [np.array([1.0, 2.0, 3.0])]
        cls.mock_encoder = Mock()
        cls.mock_encoder.generate_embeddings.return_value = [np.array([1.0, 2.0, 3.0])]
        cls.mock_encoder.reshape_embedding.return_value = np.array([[1.0, 2.0, 3.0]])


    def setUp(self):
        self.cohesive = Cohesive(model_name_or_path="mock_model")
        self.cohesive.encoder = self.mock_encoder


    def test_init(self):
        self.assertEqual(self.cohesive.encoder, self.mock_encoder)
        self.assertEqual(self.cohesive.segments, [])


    def test_get_similarity_scores_scipy(self):
        embeddings_a = [np.array([1.0, 2.0, 3.0])]
        embeddings_b = [np.array([4.0, 5.0, 6.0])]
        similarities = self.cohesive.get_similarity_scores(embeddings_a, embeddings_b, framework="scipy")
        self.assertEqual(len(similarities), 1)
        self.assertAlmostEqual(similarities[0], 0.0)


    def test_get_similarity_scores_sklearn(self):
        embeddings_a = [np.array([1.0, 2.0, 3.0])]
        embeddings_b = [np.array([4.0, 5.0, 6.0])]
        similarities = self.cohesive.get_similarity_scores(embeddings_a, embeddings_b, framework="sklearn")
        self.assertEqual(len(similarities), 1)
        self.assertAlmostEqual(similarities[0], 0.0)


    def test_get_similarity_scores_invalid_framework(self):
        with self.assertRaises(ValueError) as context:
            self.cohesive.get_similarity_scores([], [], framework="invalid_framework")

        self.assertIn("Invalid framework specified", str(context.exception))


    def test_calculate_distance_matrix(self):
        num_sentences = 3
        window_size = 2
        balanced_window = False
        results, couples = self.cohesive.calculate_distance_matrix(num_sentences, window_size, balanced_window)
        self.assertEqual(len(results), len(couples))
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(couples[0], tuple)


    def test_extract_embeddings_for_couples(self):
        couples = [(0, 1), (1, 2)]
        embeddings = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])]
        embeddings_a, embeddings_b = self.cohesive.extract_embeddings_for_couples(couples, embeddings)    
        self.assertEqual(len(embeddings_a), len(couples))
        self.assertEqual(len(embeddings_b), len(couples))
        self.assertIsInstance(embeddings_a[0], np.ndarray)
        self.assertIsInstance(embeddings_b[0], np.ndarray)


    def test_update_similarity_scores(self):
        results = [[0, 1, 0.5], [1, 2, 0.3]]   
        similarities = [0.8, 0.9]
        updated_results = self.cohesive.update_similarity_scores(results, similarities, exponential_scaling=True)
        self.assertEqual(len(updated_results), len(results))
        self.assertAlmostEqual(updated_results[0][2], 0.5 * (0.8**2))
        self.assertAlmostEqual(updated_results[1][2], 0.3 * (0.9**2))

    
    def test_update_similarity_scores_no_exponential_scaling(self):
        results = [[0, 1, 0.5], [1, 2, 0.3]]
        similarities = [0.8, 0.9]
        updated_results = self.cohesive.update_similarity_scores(results, similarities, exponential_scaling=False)
        self.assertEqual(len(updated_results), len(results))
        self.assertAlmostEqual(updated_results[0][2], 0.5 * 0.8)
        self.assertAlmostEqual(updated_results[1][2], 0.3 * 0.9)


    def test_update_similarity_scores_empty_results(self):
        results = []
        similarities = []
        updated_results = self.cohesive.update_similarity_scores(results, similarities, exponential_scaling=True)
        self.assertEqual(len(updated_results), 0)


    def test_create_similarity_graph(self):
        sentences = ["This is a test.", "Another test."]
        embeddings = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        window_size = 2
        framework = "scipy"
        balanced_window = False
        exponential_scaling = True

        with patch.object(Cohesive, 'calculate_distance_matrix', return_value=([[0, 1, 0.5], [1, 2, 0.3]], [(0, 1), (1, 2)])):
            with patch.object(Cohesive, 'extract_embeddings_for_couples', return_value=([np.array([1.0, 2.0, 3.0])], [np.array([4.0, 5.0, 6.0])])):
                with patch.object(Cohesive, 'get_similarity_scores', return_value=[0.8]):
                    with patch.object(Cohesive, 'update_similarity_scores', return_value=[[0, 1, 0.5]]):
                        result = self.cohesive.create_similarity_graph(sentences, embeddings, window_size, framework, balanced_window, exponential_scaling)
                        self.assertEqual(result, [[0, 1, 0.5]])


    def test_create_nx_graph(self):
        graph = [[0, 1, 0.5], [1, 2, 0.3]]
        nx_graph = self.cohesive.create_nx_graph(graph)
        self.assertIsInstance(nx_graph, nx.Graph)
        self.assertTrue(nx_graph.has_edge(0, 1))
        self.assertAlmostEqual(nx_graph[0][1]['weight'], 0.5)


    def test_find_best_partition(self):
        nx_graph = nx.Graph()
        nx_graph.add_edges_from([(0, 1, {'weight': 0.5}), (1, 2, {'weight': 0.3})])
        louvain_resolution = 1.0

        with patch.object(community_louvain, 'best_partition', return_value={0: 0, 1: 1, 2: 0}):
            result = self.cohesive.find_best_partition(nx_graph, louvain_resolution)
            self.assertEqual(result, {0: 0, 1: 1, 2: 0})


    def test_compact_clusters(self):
        clusters = [[0, 1, 2], [2, 3, 4]]
        result = self.cohesive.compact_clusters(clusters)
        self.assertEqual(result, [[0, 1, 2, 3, 4]])


    def test_convert_partition_to_segments(self):
        partition = {0: 0, 1: 1, 2: 0}
        result = self.cohesive.convert_partition_to_segments(partition)
        self.assertEqual(result, [[0, 2], [1]])


    def test_create_segments(self):
        sentences = ["This is a test.", "Another test."]
        window_size = 2
        louvain_resolution = 1.0
        framework = "scipy"
        show_progress_bar = False
        balanced_window = False
        exponential_scaling = True
        max_sentences_per_segment = None

        with patch.object(Cohesive, 'generate_embeddings', return_value=[np.array([1.0, 2.0, 3.0])]):
            with patch.object(Cohesive, 'create_similarity_graph', return_value=[[0, 1, 0.5]]):
                with patch.object(Cohesive, 'create_nx_graph', return_value=nx.Graph()):
                    with patch.object(Cohesive, 'find_best_partition', return_value={0: 0, 1: 1}):
                        with patch.object(Cohesive, 'convert_partition_to_segments', return_value=[[0], [1]]):
                            result = self.cohesive.create_segments(
                                sentences, window_size, louvain_resolution, framework, show_progress_bar, 
                                balanced_window, exponential_scaling, max_sentences_per_segment
                            )
                            self.assertEqual(result, "Generated 2 segments.")


    def test_get_segment_contents(self):
        segment = Segment(0, [Sentence(0, "This is a test.")])
        self.cohesive.segments = [segment]
        with patch.object(Cohesive, 'clean_text', return_value="This is a test."):
            result = self.cohesive.get_segment_contents()
            self.assertEqual(result, ["This is a test."])


    def test_print_segment_contents(self):
        segment = Segment(0, [Sentence(0, "This is a test.")])
        self.cohesive.segments = [segment]
        with patch('builtins.print') as mock_print:
            self.cohesive.print_segment_contents()
            mock_print.assert_called_once_with("This is a test.\n")


    def test_get_segment_boundaries(self):
        segment = Segment(0, [Sentence(0, "This is a test.")])
        self.cohesive.segments = [segment]
        result = self.cohesive.get_segment_boundaries()
        self.assertEqual(result, [(0, 0)])

if __name__ == '__main__':
    unittest.main()
