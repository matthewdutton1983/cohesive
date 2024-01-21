# Import project code
import cohesive.utils as utils


class Sentence:
    def __init__(self, index: int, content: str):
        """
        Initializes the Sentence class.

        Parameters:
        - index (int): The index of the sentence.
        - content (str): The content of the sentence.
        """
        self.index: int = index
        self.content: str = content


    def __repr__(self) -> str:
        """
        Returns a string representation of the Sentence object.

        Returns:
        - str: String representation of the Sentence object.
        """
        sentence_content = utils.clean_text(self.content)
        return "Sentence(index={}, content='{}')".format(self.index, sentence_content)


class Segment:
    def __init__(self, index: int, sentences: list):
        """
        Initializes the Segment class.

        Parameters:
        - index (int): The index of the segment.
        - sentences (list): List of Sentence objects representing the sentences in the segment.
        """
        self.index: int = index
        self.sentences: list = sentences


    def __repr__(self) -> str:
        """
        Returns a string representation of the Segment object.

        Returns:
        - str: String representation of the Segment object.
        """
        sentences_repr = ", ".join(repr(sentence) for sentence in self.sentences)
        return "Segment(index={}, content=[{}])".format(self.index, sentences_repr)
