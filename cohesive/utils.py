# Import standard libraries
import re
import unicodedata


def clean_text(text):
    cleaned_text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    return cleaned_text.strip()


def calculate_window_indices(i, window_size, num_sentences, balanced_window):
    start = max(0, i - (window_size // 2)) if balanced_window else i + 1
    end = min(i + 1 + (window_size // 2), num_sentences) if balanced_window else i + 1 + window_size
    
    return start, end


def extract_segment_boundaries(segment):
    start = segment.sentences[0].index
    end = segment.sentences[-1].index

    return start, end


def generate_sentence_couples(num_sentences, window_size, balanced_window):
    couples = []

    for i in range(num_sentences):
        start, end = calculate_window_indices(i, window_size, num_sentences, balanced_window)
        end = min(end, num_sentences)

        couples.extend([i, j] for j in range(start, end))

    return couples


def find_overlap(vector1, vector2):
    if not vector1 or not vector2:
        return [], 0, 0
    
    min_v1, max_v1 = min(vector1), max(vector1)
    min_v2, max_v2 = min(vector2), max(vector2)

    one_in_two = [num for num in vector1 if min_v2 <= num <= max_v2]
    two_in_one = [num for num in vector2 if min_v1 <= num <= max_v1]
    
    overlap = one_in_two + two_in_one
    
    return overlap, len(one_in_two), len(two_in_one)


def is_sequential(segments):
    for segment in segments:
        if not all(segment[i] == segment[i - 1] + 1 for i in range(1, len(segment))):
            return False
    
    return True
