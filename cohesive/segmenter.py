# Import standard libraries
import math
import re
import unicodedata
from collections import defaultdict

# Import third-party libraries
import community as community_louvain
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Import project code
from cohesive.segment import Segment
from cohesive.encoder import SentenceTransformersEncoder


class Cohesive:
    def __init__(self, model_name_or_path="paraphrase-MiniLM-L6-v2"):
        self.encoder = SentenceTransformersEncoder(model_name_or_path=model_name_or_path)
        self.segments = []


    def get_similarity_scores(self, embeddings_a, embeddings_b, framework="scipy"):
        if framework == "scipy":
            return [1 - cosine(a, b) for a, b in zip(embeddings_a, embeddings_b)]
        elif framework == "sklearn":
            embeddings_a = [self.encoder.reshape_embedding(a) for a in embeddings_a]
            embeddings_b = [self.encoder.reshape_embedding(b) for b in embeddings_b]
            return [1 - cosine_similarity(a, b) for a, b in zip(embeddings_a, embeddings_b)]
        

    def calculate_window_indices(self, i, window_size, num_sentences, balanced_window):
        start = max(0, i - (window_size // 2)) if balanced_window else i + 1
        end = min(i + 1 + (window_size // 2), num_sentences) if balanced_window else i + 1 + window_size
        
        return start, end


    def generate_sentence_couples(self, num_sentences, window_size, balanced_window):
        couples = []

        for i in range(num_sentences):
            start, end = self.calculate_window_indices(i, window_size, num_sentences, balanced_window)
            end = min(end, num_sentences)
    
            couples.extend([i, j] for j in range(start, end))

        return couples


    def calculate_distance_matrix(self, num_sentences, window_size, balanced_window):
        results = []
        couples = self.generate_sentence_couples(num_sentences, window_size, balanced_window)

        for i, j in couples:
            results.append([i, j, math.exp(-abs(j - i) / 2)])
    
        return results, couples
    

    def extract_embeddings_for_couples(self, couples, embeddings):
        a, b = zip(*couples)

        embeddings_a = [embeddings[i] for i in a]
        embeddings_b = [embeddings[i] for i in b]

        return embeddings_a, embeddings_b


    def update_similarity_scores(self, results, similarities, exponential_scaling):
        for i, s in enumerate(similarities):
            results[i][2] *= (s**2) if exponential_scaling else s

        return results
    

    def create_similarity_graph(
            self, 
            sentences, 
            embeddings, 
            window_size, 
            framework, 
            balanced_window, 
            exponential_scaling
        ):
        num_sentences = len(sentences)
        results, couples = self.calculate_distance_matrix(num_sentences, window_size, balanced_window)
        embeddings_a, embeddings_b = self.extract_embeddings_for_couples(couples, embeddings)
        similarities = self.get_similarity_scores(embeddings_a, embeddings_b, framework)
        similarity_graph = self.update_similarity_scores(results, similarities, exponential_scaling)
        
        return similarity_graph
    

    def create_nx_graph(self, graph):
        G = nx.Graph()

        for node in graph:
            G.add_edge(node[0], node[1], weight=node[2])

        return G


    def find_best_partition(self, nx_graph, louvain_resolution):
        return community_louvain.best_partition(
            nx_graph, 
            resolution=louvain_resolution, 
            weight="weight", 
            randomize=False, 
            random_state=256
        )
                        

    def find_overlap(self, vector1, vector2):
        if not vector1 or not vector2:
            return [], 0, 0
        
        min_v1, max_v1 = min(vector1), max(vector1)
        min_v2, max_v2 = min(vector2), max(vector2)

        one_in_two = [num for num in vector1 if min_v2 <= num <= max_v2]
        two_in_one = [num for num in vector2 if min_v1 <= num <= max_v1]
        
        overlap = one_in_two + two_in_one
        
        return overlap, len(one_in_two), len(two_in_one)


    def compact_clusters(self, clusters):
        compact_clusters = []
        
        while clusters:
            curr_cl = clusters.pop(0)
            
            for target_cl in clusters[:]:
                overlap, n_1_in_2, n_2_in_1 = self.find_overlap(target_cl, curr_cl)
            
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


    def convert_partition_to_segments(self, partition):
        raw_segments = defaultdict(list)

        for k, v in partition.items():
            raw_segments[v].append(k)

        segment_indices = self.compact_clusters(list(raw_segments.values()))

        return segment_indices
    

    def clean_text(self, text):
        cleaned_text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text.strip()


    def is_sequential(self, segments):
        for segment in segments:
            if not all(segment[i] == segment[i - 1] + 1 for i in range(1, len(segment))):
                return False
        
        return True
    

    def extract_segment_boundaries(self, segment):
        start = segment.sentences[0][0]
        end = segment.sentences[-1][0]

        return start, end
         

    def create_segments(
            self, 
            sentences, 
            window_size=4, 
            louvain_resolution=1.,
            framework="scipy",
            show_progress_bar=False, 
            balanced_window=False, 
            exponential_scaling=False,
            max_sentences_per_segment=None
        ):
        self.segments = []

        embeddings = self.encoder.generate_embeddings(sentences, show_progress_bar)
        similarity_graph = self.create_similarity_graph(sentences, embeddings, window_size, framework, balanced_window, exponential_scaling)
        nx_graph = self.create_nx_graph(similarity_graph)
        partition = self.find_best_partition(nx_graph, louvain_resolution)
        segment_indices = self.convert_partition_to_segments(partition)

        current_segment_index = 0
        for seg in segment_indices:
            if max_sentences_per_segment is not None and len(seg) > max_sentences_per_segment:
                sub_segments = [seg[i:i + max_sentences_per_segment] for i in range(0, len(seg), max_sentences_per_segment)]
                
                for sub_seg in sub_segments:
                    self.segments.append(
                        Segment(current_segment_index, [(sent_idx, sentences[sent_idx]) for sent_idx in sorted(sub_seg)])
                    )
                    current_segment_index += 1
            else:
                self.segments.append(
                    Segment(current_segment_index, [(sent_idx, sentences[sent_idx]) for sent_idx in sorted(seg)])
                )
                current_segment_index += 1

        if not self.is_sequential(segment_indices):
            print("Sentence order not preserved.")

        return f"Generated {len(self.segments)} segments."


    def get_segment_contents(self):       
        segment_contents = [
            " ".join(self.clean_text(sent) for _, sent in segment.sentences)
            for segment in self.segments
        ]
        return segment_contents
            

    def get_segment_boundaries(self):
        return [self.extract_segment_boundaries(segment) for segment in self.segments]
    