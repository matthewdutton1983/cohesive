# Import standard libraries
import unittest
from unittest.mock import Mock, patch

# Import project code
from cohesive.base import Sentence, Segment


class TestUtils(unittest.TestCase):
    def test_sentence_initialization(self):
        sentence = Sentence(index=1, content="This is a test sentence.")
        self.assertEqual(sentence.index, 1)
        self.assertEqual(sentence.content, "This is a test sentence.")


    def test_sentence_representation(self):
        sentence = Sentence(index=1, content="This is a test sentence.")
        expected_repr = "Sentence(index=1, content='this is a test sentence')"
        self.assertEqual(repr(sentence), expected_repr)


    def test_segment_initialization(self):
        sentence1 = Mock(index=1, content="First sentence.")
        sentence2 = Mock(index=2, content="Second sentence.")
        sentences = [sentence1, sentence2]
        segment = Segment(index=1, sentences=sentences)

        self.assertEqual(segment.index, 1)
        self.assertEqual(segment.sentences, sentences)


    def test_segment_representation(self):
        sentence1 = Mock(index=1, content="First sentence.")
        sentence2 = Mock(index=2, content="Second sentence.")
        sentences = [sentence1, sentence2]
        segment = Segment(index=1, sentences=sentences)

        expected_repr = "Segment(index=1, content=[{}, {}])".format(repr(sentence1), repr(sentence2))
        self.assertEqual(repr(segment), expected_repr)


    @patch('your_module.utils.clean_text', return_value="Cleaned content")
    def test_sentence_representation_with_cleaning(self, mock_clean_text):
        sentence = Sentence(index=1, content="Dirty content.")
        expected_repr = "Sentence(index=1, content='cleaned content')"
        self.assertEqual(repr(sentence), expected_repr)
        mock_clean_text.assert_called_once_with("Dirty content.")

if __name__ == '__main__':
    unittest.main()
