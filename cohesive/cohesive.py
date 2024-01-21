# Import standard libraries
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union

# Import third-party libraries
import community as community_louvain
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Import project code
import cohesive.utils as utils
from cohesive.base import Segment, Sentence
from cohesive.encoder import SentenceTransformersEncoder


class Cohesive:
    def __init__(self, model_name_or_path: str = "paraphrase-MiniLM-L6-v2"):
        """
        Initializes the Cohesive class.

        Parameters:
        - model_name_or_path (str): The name or path of the Sentence Transformers model. Default is "paraphrase-MiniLM-L6-v2".
        """
        self.encoder: SentenceTransformersEncoder = SentenceTransformersEncoder(model_name_or_path=model_name_or_path)
        self.segments: List[Segment] = []


    def get_similarity_scores(
            self, 
            embeddings_a: List[np.ndarray], 
            embeddings_b: List[np.ndarray], 
            framework: str = "scipy"
        ) -> List[float]:
        """
        Calculate similarity scores between two sets of embeddings using the specified framework.

        Parameters:
        - embeddings_a (List[np.ndarray]): List of embeddings for set A.
        - embeddings_b (List[np.ndarray]): List of embeddings for set B.
        - framework (str): The framework to use for calculating similarity scores. Default is "scipy".

        Returns:
        - List[float]: List of similarity scores.
        """
        frameworks = ["scipy", "sklearn"]

        if framework == "scipy":
            return [1 - cosine(a, b) for a, b in zip(embeddings_a, embeddings_b)]
        elif framework == "sklearn":
            embeddings_a = [self.encoder.reshape_embedding(a) for a in embeddings_a]
            embeddings_b = [self.encoder.reshape_embedding(b) for b in embeddings_b]
            return [1 - cosine_similarity(a, b)[0][0] for a, b in zip(embeddings_a, embeddings_b)]
        else:
            raise ValueError("Invalid framework specified. Choose from: {}".format(frameworks))


    def calculate_distance_matrix(
            self, 
            num_sentences: int, 
            window_size: int, 
            balanced_window: bool
        ) -> Tuple[List[List[Union[int, int, float]]], List[Tuple[int, int]]]:
        """
        Calculate a distance matrix and generate sentence couples.

        Parameters:
        - num_sentences (int): The total number of sentences.
        - window_size (int): The window size for generating couples.
        - balanced_window (bool): Flag to use a balanced window.

        Returns:
        - Tuple[List[List[Union[int, int, float]]], List[Tuple[int, int]]]: Results and couples.
        """
        results = []
        couples = utils.generate_sentence_couples(num_sentences, window_size, balanced_window)

        for i, j in couples:
            results.append([i, j, math.exp(-abs(j - i) / 2)])

        return results, couples
    

    def extract_embeddings_for_couples(
          self, 
          couples: List[Tuple[int, int]], 
          embeddings: List[np.ndarray]
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract embeddings for couples.

        Parameters:
        - couples (List[Tuple[int, int]]): List of couples (indices).
        - embeddings (List[np.ndarray]): List of embeddings.

        Returns:
        - Tuple[List[np.ndarray], List[np.ndarray]]: Embeddings for set A and set B.
        """
        a, b = zip(*couples)

        embeddings_a = [embeddings[i] for i in a]
        embeddings_b = [embeddings[i] for i in b]

        return embeddings_a, embeddings_b


    def update_similarity_scores(
            self, 
            results: List[List[Union[int, int, float]]], 
            similarities: List[float], 
            exponential_scaling: bool
        ) -> List[List[Union[int, int, float]]]:
        """
        Update similarity scores in the results.

        Parameters:
        - results (List[List[Union[int, int, float]]]): List of results.
        - similarities (List[float]): List of similarity scores.
        - exponential_scaling (bool): Flag to use exponential scaling.

        Returns:
        - List[List[Union[int, int, float]]]: Updated results.
        """
        for i, s in enumerate(similarities):
            results[i][2] *= (s**2) if exponential_scaling else s

        return results
    

    def create_similarity_graph(
            self, 
            sentences: List[str], 
            embeddings: List[np.ndarray], 
            window_size: int, 
            framework: str, 
            balanced_window: bool, 
            exponential_scaling: bool
        ) -> List[List[Union[int, int, float]]]:
        """
        Create a similarity graph based on the input sentences and embeddings.

        Parameters:
        - sentences (List[str]): List of sentences.
        - embeddings (List[np.ndarray]): List of embeddings.
        - window_size (int): The window size for generating couples.
        - framework (str): The framework to use for calculating similarity scores.
        - balanced_window (bool): Flag to use a balanced window.
        - exponential_scaling (bool): Flag to use exponential scaling.

        Returns:
        - List[List[Union[int, int, float]]]: Similarity graph.
        """
        num_sentences = len(sentences)
        results, couples = self.calculate_distance_matrix(num_sentences, window_size, balanced_window)
        embeddings_a, embeddings_b = self.extract_embeddings_for_couples(couples, embeddings)
        similarities = self.get_similarity_scores(embeddings_a, embeddings_b, framework)
        similarity_graph = self.update_similarity_scores(results, similarities, exponential_scaling)

        return similarity_graph
    

    def create_nx_graph(self, graph: List[List[Union[int, int, float]]]) -> nx.Graph:
        """
        Create a NetworkX graph from the input graph.

        Parameters:
        - graph (List[List[Union[int, int, float]]]): The input graph.

        Returns:
        - nx.Graph: The NetworkX graph.
        """
        G = nx.Graph()

        for node in graph:
            G.add_edge(node[0], node[1], weight=node[2])

        return G


    def find_best_partition(
            self, 
            nx_graph: nx.Graph, 
            louvain_resolution: float
        ) -> Dict[int, int]:
        """
        Find the best partition using Louvain community detection.

        Parameters:
        - nx_graph (nx.Graph): The input NetworkX graph.
        - louvain_resolution (float): The resolution parameter for Louvain.

        Returns:
        - Dict[int, int]: A dictionary mapping nodes to their respective communities.
        """
        return community_louvain.best_partition(
            nx_graph, 
            resolution=louvain_resolution, 
            weight="weight", 
            randomize=False, 
            random_state=256
        )


    def compact_clusters(self, clusters: List[List[int]]) -> List[List[int]]:
        """
        Compact overlapping clusters.

        Parameters:
        - clusters (List[List[int]]): List of clusters.

        Returns:
        - List[List[int]]: Compact clusters.
        """
        compact_clusters = []

        while clusters:
            curr_cl = clusters.pop(0)

            for target_cl in clusters[:]:
                overlap, n_1_in_2, n_2_in_1 = utils.find_overlap(target_cl, curr_cl)

                if overlap:
                    if n_1_in_2 < n_2_in_1 or n_2_in_1 == 0:
                        curr_cl.extend(overlap)
                        curr_cl = list(set(curr_cl))
                        target_cl = list(set(target_cl) - set(overlap))
                    else:
                        target_cl.extend(overlap)
                        target_cl = list(set(target_cl))
                        curr_cl = list(set(curr_cl) - set(overlap))

                    if not curr_cl:
                        break

            if curr_cl:
                compact_clusters.append(sorted(set(curr_cl)))

        compact_clusters.sort()

        return compact_clusters


    def convert_partition_to_segments(
            self, 
            partition: Dict[int, int]
        ) -> List[List[int]]:
        """
        Convert Louvain partition to segments.

        Parameters:
        - partition (Dict[int, int]): Louvain partition.

        Returns:
        - List[List[int]]: List of segment indices.
        """
        raw_segments = defaultdict(list)

        for k, v in partition.items():
            raw_segments[v].append(k)

        segment_indices = self.compact_clusters(list(raw_segments.values()))

        return segment_indices
         

    def create_segments(
            self, 
            sentences: List[str], 
            window_size: int = 4, 
            louvain_resolution: float = 1.,
            framework: str = "scipy",
            show_progress_bar: bool = False, 
            balanced_window: bool = False, 
            exponential_scaling: bool = False,
            max_sentences_per_segment: Optional[int] = None
        ) -> str:
        """
        Create segments based on input sentences.

        Parameters:
        - sentences (List[str]): List of sentences.
        - window_size (int): The window size for generating couples. Default is 4.
        - louvain_resolution (float): The resolution parameter for Louvain. Default is 1.
        - framework (str): The framework to use for calculating similarity scores. Default is "scipy".
        - show_progress_bar (bool): Flag to show a progress bar. Default is False.
        - balanced_window (bool): Flag to use a balanced window. Default is False.
        - exponential_scaling (bool): Flag to use exponential scaling. Default is False.
        - max_sentences_per_segment (Optional[int]): Maximum number of sentences per segment. Default is None.

        Returns:
        - str: A message indicating the number of segments generated.
        """
        self.segments = []

        embeddings = self.encoder.generate_embeddings(sentences, show_progress_bar)
        similarity_graph = self.create_similarity_graph(
            sentences, embeddings, window_size, framework, balanced_window, exponential_scaling
        )
        nx_graph = self.create_nx_graph(similarity_graph)
        partition = self.find_best_partition(nx_graph, louvain_resolution)
        segment_indices = self.convert_partition_to_segments(partition)

        current_segment_index = 0

        for seg in segment_indices:
            if max_sentences_per_segment is not None and len(seg) > max_sentences_per_segment:
                sub_segments = [
                    seg[i : i + max_sentences_per_segment]
                    for i in range(0, len(seg), max_sentences_per_segment)
                ]

                for sub_seg in sub_segments:
                    self.segments.append(
                        Segment(
                            current_segment_index,
                            [
                                Sentence(sent_idx, sentences[sent_idx])
                                for sent_idx in sorted(sub_seg)
                            ],
                        )
                    )
                    current_segment_index += 1
            else:
                self.segments.append(
                    Segment(
                        current_segment_index,
                        [
                            Sentence(sent_idx, sentences[sent_idx])
                            for sent_idx in sorted(seg)
                        ],
                    )
                )
                current_segment_index += 1

        if not utils.is_sequential(segment_indices):
            print("Sentence order not preserved.")

        return f"Generated {len(self.segments)} segments."


    def get_segment_contents(self) -> List[str]:
        """
        Get the contents of each segment.

        Returns:
        - List[str]: List of segment contents.
        """
        segment_contents = [
            " ".join(utils.clean_text(sent.content) for sent in segment.sentences)
            for segment in self.segments
        ]

        return segment_contents
    

    def print_segment_contents(self) -> str:
        """
        Prints the contents of each segment.
        
        Returns:
        - str: Content of each segment.
        """
        segments = self.get_segment_contents()

        for segment in segments:
            print(segment.replace("\n\n", "\n").replace("\n", " ") + "\n")
            

    def get_segment_boundaries(self) -> List[Tuple[int, int]]:
        """
        Get the boundaries of each segment.

        Returns:
        - List[Tuple[int, int]]: List of segment boundaries.
        """
        return [utils.extract_segment_boundaries(segment) for segment in self.segments]
    