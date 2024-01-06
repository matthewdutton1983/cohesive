# cohesive

cohesive is a lightweight segmenter that uses sentence embeddings to split documents into naturally coherent segments akin to paragraphs.

## Installation

You can install 'cohesive' using pip:

```bash
pip install cohesive
```

## Using cohesive

To start using the SDK, simply import the CohesiveTextSegmenter and create a new instance:

```python
from cohesive import CohesiveTextSegmenter

# Create a new instance of semantify by passing in the name of the SentenceTransformer model that you want to use to generate the chunks.
# By default, cohesive utilizes paraphrase-MiniLM-L6-v2, which has produced good results.
cohesive = CohesiveTextSegmenter("all-MiniLM-L6-v2")

# Then, all you need to do is call the generate_tiles method and pass in an array of sentences.
cohesive.generate_segments(sentences)
```
