# cohesive

cohesive is a lightweight segmenter that uses sentence embeddings to split documents into naturally coherent segments akin to paragraphs.

## Installation

You can install cohesive using pip:

```bash
pip install cohesive
```

## Using cohesive

To start using cohesive, import Cohesive and the relevant text embedding class. Choose from OpenAI, SentenceTransformers, Tensorflow, or Transformers:

```python
from cohesive import Cohesive

# By default, cohesive uses the paraphrase-MiniLM-L6-v2 model, which produces good
# results, but you can pass the name of any model into the Cohesive constructor.
cohesive = Cohesive("msmarco-distilbert-cos-v5")

# Then, all you need to do is call the create_segments method and pass in an
# array of sentences.
cohesive.create_segments(sentences)
```

## Finetuning cohesive

cohesive users can finetune several parameters, which all impact the final segmentation results in different ways. Here is a quick summary:

- **window_size:** Sets the size of the context window for generating segments. Defaults to 4.
- **louvain_resolution:** Used by the Louvain community detection algorithm to partition sentences into segments. Default is 1.
- **framework:** The framework to use for calculating similarity scores. Choose between scipy and sklearn. Default is "scipy".
- **show_progress_bar:** Flag to display the progress bar from sentence-transformers whilst generating embeddings. Defaults to False.
- **balanced_window:** If True, the context window is split evenly between preceding and subsequent sentences, otherwise it only looks at subsequent sentences. Defaults to False.
- **exponential_scaling:** Flag to use exponential scaling when calculating similarity scores. Defaults to False.
- **max_sentences_per_segment:** Maximum number of sentences per segment. Default is None.

To modify the parameters, simply pass in the appropriate parameter name and value when you call the create_segments method:

```python
# Via create_segments
cohesive.create_segments(sentences, window_size=3, exponential_scaling=True)
```

## Viewing the segments

When **create_segments** has finished, cohesive will print a summary of the total number of segments that were created.

There are several methods for interacting with the generated segments.

```python
# View a string representation of the consolidated Segment and Sentence objects
cohesive.segments

# List that contains the content of each segment.
cohesive.get_segment_contents()

# View the start and end indices of sentences within a segment.
cohesive.get_segment_boundaries()

# Print the contents of each segment to the console or Notebook.
cohesive.print_segment_contents()
```
