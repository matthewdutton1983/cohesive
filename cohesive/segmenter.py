# Import standard libraries
import re
import unicodedata
from collections import defaultdict

# Import third-party libraries
import community as community_louvain
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

ALPHA = 0.5
CONTEXT_WINDOW = 8
DECAY = 0.8
RESOLUTION = 1.0
  

class Document:
    def __init__(self, segments):
        self.segments = segments

    def __repr__(self):
        return "Document({})".format(self.segments)
        

class Segment:
    def __init__(self, index, sentences):
        self.index = index
        self.sentences = sentences


    def __repr__(self):
        return "Segment({})".format(self.sentences)
    

class Sentence:
    def __init__(self, index, text):
        self.index = index
        self.text = text


    def __repr__(self):
        return "Sentence({})".format(self.text)


class CohesiveTextSegmenter:


    def __init__(self, model_name_or_path="paraphrase-MiniLM-L6-v2"):
        """Initializes the CohesiveTextSegmenter."""
        self.model = self._load_model(model_name_or_path)
        self.alpha = ALPHA
        self.context_window = CONTEXT_WINDOW
        self.decay = DECAY
        self.resolution = RESOLUTION
        self.segments = None


    def _load_model(self, model_name_or_path):
        """Loads the model from the specified path."""
        try:  
          return SentenceTransformer(model_name_or_path)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}")
            

    def _generate_embeddings(self, sentences):
        """Generates embeddings for the a list of text chunks."""    
        return self.model.encode(sentences, show_progress_bar=True)
            
        
    def _normalize_similarities(self, similarity_matrix, embeddings):
        """Normalizes the similarity scores for cosine similarity."""
        return similarity_matrix / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]


    def _create_similarity_matrix(self, embeddings):
        """Creates a similarity matrix for all embeddings."""
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return self._normalize_similarities(similarity_matrix, embeddings)


    def _calculate_local_similarities(self, sentences, similarity_matrix):
        """Calculates a similarity matrix for all embeddings within the context window."""
        local_similarities = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentences)):
            start = max(0, i - self.context_window)
            end = min(len(sentences), i + self.context_window + 1)

            local_similarities[i, start:end] = similarity_matrix[i, start:end]
        
        return local_similarities


    def _calculate_combined_similarities(self, sentences, similarity_matrix, local_similarities):
        """Combines global and local similarities via a weighted average."""
        combined_similarities = self.alpha * similarity_matrix + (1 - self.alpha) * local_similarities

        indices = np.triu_indices(len(sentences), k=1)
        distances = np.abs(indices[0] - indices[1])
        decay_factors = np.exp(-self.decay * distances)

        combined_similarities[indices] *= decay_factors
        combined_similarities[indices[::-1]] *= decay_factors
        
        return combined_similarities


    def _create_nx_graph(self, combined_similarities):
        """Builds a NetworkX graph from the combined similarities."""
        return nx.from_numpy_array(combined_similarities)


    def _find_best_partition(self, nx_graph):
        """Finds community partitions in a NetworkX graph."""
        return community_louvain.best_partition(nx_graph, resolution=self.resolution, weight="weight", randomize=False)
    

    def _merge_clusters(self, clusters):
        """Iteratively merges overlapping clusters."""
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


    def _group_nodes_into_segments(self, partition):
        """Groups nodes into segments based on community detection."""
        segments = defaultdict(list)

        for node, community in partition.items():
            segments[community].append(node)
        
        sorted_segments = [sorted(seg) for seg in segments.values()]
        merged_clusters = self._merge_clusters(sorted_segments)

        return merged_clusters
    

    def _extract_segment_boundaries(self, segment):
        """Extracts the start and end indices of sentences in a segment."""
        start_index = segment[1][0][0]
        end_index = segment[1][-1][0]
        
        return start_index, end_index
    

    def _clean_text(self, text):
        """Removes unicode characters and duplicate whitespace."""
        cleaned_text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        
        return cleaned_text.strip()
    

    def _update_parameters(self, **kwargs):
        """Updates the specified parameters."""
        valid_params = {"alpha", "context_window", "decay", "resolution"}

        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError("Invalid parameter specfied. Choose from: {}".format(valid_params))
    

    def create_segments(self, sentences, **kwargs):
        """Creates naturally coherent segments by grouping semantically similar sentences."""
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


    def print_segments(self):
        """Prints the contents of each segment to the console."""
        for _, sentences in self.segments:
            joined_sentences = " ".join(self._clean_text(sent) for _, sent in sentences)
            print(joined_sentences + "\n")

    
    def print_segment_boundaries(self):
        """Prints the start and end indices of sentences within a segment."""
        for segment in self.segments:
            print(self._extract_segment_boundaries(segment))


    def get_params(self):
        """Displays a summary of the current parameter values."""
        return {
            "alpha": self.alpha,
            "context_window": self.context_window,
            "decay": self.decay,
            "resolution": self.resolution
        }

    
    def finetune_params(self, **kwargs):
        """Allows the user to dynamically change the parameters as needed."""
        return self._update_parameters(**kwargs)
