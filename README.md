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
from cohesive.embeddings import SentenceTransformersEmbeddings

# Instantiate your embedding of choice and pass it into the Cohesive constructor.
# Each embedding class is configured to run a default model.
# For example, SentenceTransformersEmbeddings runs all-Mini-LM-L6-v2 out of the box.
# But you can pass the name of any model into the embeddings constructor.
embeddings = SentenceTransformersEmbeddings("msmarco-distilbert-cos-v5")
cohesive = Cohesive(embedding)

# Then, all you need to do is call the create_segments method and pass
# in an array of sentences.
cohesive.create_segments(sentences)
```

## Finetuning cohesive

cohesive users can finetune several parameters, which all impact the final segmentation results in different ways. Here is a quick summary:

- **Alpha**:

  - **Role**: Used as a weight in combining global and local similarities.
  - **Impact**: A higher alpha places more emphasis on global similarities, making the segmentation more influenced by overall similarity between sentences. Conversely, a lower alpha gives more weight to local similarities within the context window.
  - **Default**: 0.5

- **Decay**:

  - **Role**: Used to calculate decay factors based on distances between sentence indices when combining similarities.
  - **Impact**: A higher decay results in faster decay with increasing distance between sentences. This means that sentences further apart contribute less to the overall similarity, emphasizing local cohesion. A lower decay allows for longer-range dependencies to impact the segmentation.
  - **Default**: 0.8

- **Resolution**:

  - **Role**: Used in the community detection algorithm.
  - **Impact**: A higher resolution value leads to more and smaller communities, potentially yielding finer-grained segmentation. Conversely, a lower resolution results in fewer and larger communities, offering a more consolidated segmentation.
  - **Default**: 1.0

- **Window Direction**:

  - **Role**: Controls the directionality of the context window when calculating local similarities.
  - **Impact**: Determines whether the context window is symmetrical or asymmetrical, focusing on preceeding or subsequent sentences, or both.
  - **Default**: "bidirectional"

- **Window Size**:

  - **Role**: Determines the size of the context window used to calculate local similarities between sentences.
  - **Impact**: A smaller context window focuses on very close neighbors, capturing fine-grained local relationships. This may be suitable for documents where coherence is established within a small span of sentences. On the other hand, a larger context window considers a broader context, capturing longer-range dependencies and global patterns.
  - **Default**: 6

To modify the parameters, either pass in the appropriate parameter name and value when you call the create_segments method, or use the dedicated finetune_params function:

```python
# Via create_segments
cohesive.create_segments(sentences, window_size=3)

# Via finetune_params
cohesive.finetune_params(alpha=1, decay=0.2)
```

**Note: Any update to the parameters is stateful.**

At any time you can view the current parameters by calling the **get_params** method.

## Viewing the segments

When **create_segments** has finished, cohesive will print a summary of the total number of segments that were created.

To view the segments, simply call the **print_segments** method.

You can also view the start and end indices of sentences with a segment via the **print_segment_boundaries** function.
