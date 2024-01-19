class Segment:
  def __init__(self, index, sentences):
    self.index = index
    self.sentences = sentences
    self.total_characters = self.calculate_segment_length()
    self.total_sentences = len(sentences)


  def calculate_segment_length(self):
    return sum(len(sentence[1]) for sentence in self.sentences)
  

  def __repr__(self):
    return f"Segment(index={self.index}, sentences={self.sentences}, total_sentences={self.total_sentences}, total_characters={self.total_characters})"
  