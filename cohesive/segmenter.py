# Import standard libraries
import re
import unicodedata
from collections import defaultdict

# Import third-party libraries
import community as community_louvain
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

ALPHA: float = 0.5
CONTEXT_WINDOW: int = 6
DECAY: float = 0.8
RESOLUTION: float = 1.0  


class CohesiveTextSegmenter:
    def __init__(self, model_name_or_path: str = "paraphrase-MiniLM-L6-v2", **kwargs):
        """Initializes the CohesiveTextSegmenter.

        Args:
            model_name_or_path: Path to the sentence transformer model.
        """
        self.model = self._load_model(model_name_or_path, **kwargs)
        self.alpha: float = ALPHA
        self.context_window: int = CONTEXT_WINDOW
        self.decay: float = DECAY
        self.resolution: float = RESOLUTION
        self.segments: list[tuple[int, list[tuple[int, str]]]] = None


    def _load_model(self, model_name_or_path: str, **kwargs) -> SentenceTransformer:
        """Loads the model from the specified path.

        Args:
            model_name_or_path: Path to the sentence transformer model.
            **kwargs: Additional keyword arguments to pass to the SentenceTransformer constructor.

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            ValueError: If there's an error loading the model.
        """
        try:  
          return SentenceTransformer(model_name_or_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}")
            

    def _generate_embeddings(self, sentences: list[str]) -> np.ndarray:
        """Generates embeddings for the a list of text chunks.

        Args:
            sentences: A list of sentences to embed.

        Returns:
            A NumPy array of sentence embeddings.
        """    
        return self.model.encode(sentences, show_progress_bar=True)
            
        
    def _normalize_similarities(self, similarity_matrix: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Normalizes the similarity scores for cosine similarity.

        Args:
            similarity_matrix: The similarity matrix to normalize.
            embeddings: The embeddings used to compute the similarity matrix.

        Returns:
            The normalized similarity matrix.
        """
        return similarity_matrix / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]


    def _create_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Creates a similarity matrix for all embeddings.

        Args:
            embeddings: A NumPy array of sentence embeddings.

        Returns:
            A NumPy array representing the similarity matrix.
        """
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return self._normalize_similarities(similarity_matrix, embeddings)


    def _calculate_local_similarities(self, sentences: list[str], similarity_matrix: np.ndarray) -> np.ndarray:
        """Calculates a similarity matrix for all embeddings within the context window.

        Args:
            sentences: A list of sentences.
            similarity_matrix: The global similarity matrix.

        Returns:
            A NumPy array representing the local similarity matrix.
        """
        local_similarities = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentences)):
            start = max(0, i - self.context_window)
            end = min(len(sentences), i + self.context_window + 1)

            local_similarities[i, start:end] = similarity_matrix[i, start:end]
        
        return local_similarities


    def _calculate_combined_similarities(
            self, 
            sentences: list[str], 
            similarity_matrix: np.ndarray, 
            local_similarities: np.ndarray
        ) -> np.ndarray:
        """Combines global and local similarities via a weighted average."""
        combined_similarities = self.alpha * similarity_matrix + (1 - self.alpha) * local_similarities

        indices = np.triu_indices(len(sentences), k=1)
        distances = np.abs(indices[0] - indices[1])
        decay_factors = np.exp(-self.decay * distances)

        combined_similarities[indices] *= decay_factors
        combined_similarities[indices[::-1]] *= decay_factors
        
        return combined_similarities


    def _create_nx_graph(self, combined_similarities: np.ndarray) -> nx.Graph:
        """Builds a NetworkX graph from the combined similarities.

        Args:
            combined_similarities: A NumPy array representing the combined similarity matrix.

        Returns:
            A NetworkX graph representing the sentence relationships.
        """
        return nx.from_numpy_array(combined_similarities)

    def _find_best_partition(self, nx_graph: nx.Graph) -> dict[int, int]:
        """Finds community partitions in a NetworkX graph.

        Args:
            nx_graph: The NetworkX graph to analyze.

        Returns:
            A dictionary mapping nodes to their assigned community.
        """
        return community_louvain.best_partition(nx_graph, resolution=self.resolution, weight="weight", randomize=False)

    def _merge_clusters(self, clusters: list[np.ndarray]) -> list[np.ndarray]:
        """Iteratively merges overlapping clusters.

        Args:
            clusters: A list of NumPy arrays representing clusters.

        Returns:
            A list of merged clusters represented by NumPy arrays.
        """
        merged_clusters = []

        while clusters:
            current_cluster = clusters.pop(0)

            for next_cluster in clusters:
                if np.any(np.isin(current_cluster, next_cluster)):
                    current_cluster = np.union1d(current_cluster, next_cluster)
                    clusters.remove(next_cluster)
                    break

            merged_clusters.append(current_cluster)

        return merged_clusters

    def _group_nodes_into_segments(self, partition: dict[int, int]) -> list[list[int]]:
        """Groups nodes into segments based on community detection.

        Args:
            partition: A dictionary mapping nodes to their assigned community.

        Returns:
            A list of lists containing the indices of nodes within each segment.
        """
        segments = defaultdict(list)

        for node, community in partition.items():
            segments[community].append(node)

        sorted_segments = [sorted(seg) for seg in segments.values()]
        merged_clusters = self._merge_clusters(sorted_segments)

        return merged_clusters

    def _extract_segment_boundaries(self, segment: list[tuple[int, str]]) -> tuple[int, int]:
        """Extracts the start and end indices of sentences in a segment.

        Args:
            segment: A list of tuples containing sentence indices and texts.

        Returns:
            A tuple representing the start and end indices of the segment.
        """
        start_index = segment[1][0][0]
        end_index = segment[1][-1][0]

        return start_index, end_index

    def _clean_text(self, text: str) -> str:
        """Removes unicode characters and duplicate whitespace.

        Args:
            text: The text to clean.

        Returns:
            The cleaned text.
        """
        cleaned_text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text.strip()

    def _update_parameters(self, **kwargs):
        """Updates the specified parameters.

        Args:
            **kwargs: Keyword arguments containing valid parameter names and values.

        Raises:
            ValueError: If an invalid parameter is specified.
        """
        valid_params = {"alpha", "context_window", "decay", "resolution"}

        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError("Invalid parameter specfied. Choose from: {}".format(valid_params))
    

    def create_segments(self, sentences: list[str], **kwargs) -> str:
        """Creates naturally coherent segments by grouping semantically similar sentences.

        Args:
            sentences: A list of sentences to segment.
            **kwargs: Keyword arguments to update specific parameters.

        Returns:
            A string indicating the number of generated segments.

        Raises:
            ValueError: If the input sentence list is empty.
        """
        if not sentences:
            raise ValueError("Input sentences list is empty.")
        
        self._update_parameters(**kwargs)
        
        embeddings = self._generate_embeddings(sentences)

        if len(sentences) != len(embeddings):
            raise ValueError("The number of sentences and embeddings does not match.")

        similarity_matrix = self._create_similarity_matrix(embeddings)
        local_similarities = self._calculate_local_similarities(sentences, similarity_matrix)
        combined_similarities = self._calculate_combined_similarities(sentences, similarity_matrix, local_similarities)
        nx_graph = self._create_nx_graph(combined_similarities)
        partition = self._find_best_partition(nx_graph)
        segment_indices = self._group_nodes_into_segments(partition)
    
        self.segments = [(idx, [(sent_idx, sentences[sent_idx]) for sent_idx in seg]) for idx, seg in enumerate(segment_indices)]

        return f"Generated {len(self.segments)} segments."


    def print_segments(self) -> None:
        """Prints the contents of each segment to the console."""
        for _, sentences in self.segments:
            joined_sentences = " ".join(self._clean_text(sent) for _, sent in sentences)
            print(joined_sentences + "\n")

    
    def print_segment_boundaries(self) -> None:
        """Prints the start and end indices of sentences within a segment."""
        for segment in self.segments:
            print(self._extract_segment_boundaries(segment))
            

    def get_params(self) -> dict[str, float]:
        """Displays a summary of the current parameter values.

        Returns:
          A dictionary containing parameter names and their current values.
        """
        return {
            "alpha": self.alpha,
            "context_window": self.context_window,
            "decay": self.decay,
            "resolution": self.resolution
        }

    
    def finetune_params(self, **kwargs) -> None:
        """Allows the user to dynamically change the parameters as needed.

        Args:
            **kwargs: Keyword arguments containing valid parameter names and values.

        Raises:
            ValueError: If an invalid parameter is specified.
        """
        return self._update_parameters(**kwargs)
    
