# cohesive

cohesive is a lightweight segmenter that uses sentence embeddings to split documents into naturally coherent segments akin to paragraphs.

## Installation

You can install cohesive using pip:

```bash
pip install cohesive
```

## Using cohesive

To start using cohesive, import the CohesiveTextSegmenter and create a new instance:

```python
from cohesive import CohesiveTextSegmenter

# By default, cohesive uses paraphrase-MiniLM-L6-v2, which produces good
# results, but you can specify any SentenceTransformer model.
# For example, lets use all-MiniLM-L6-v2 ...
cohesive = CohesiveTextSegmenter("all-MiniLM-L6-v2")

# Then, all you need to do is call the generate_tiles method and pass
# in an array of sentences.
cohesive.generate_segments(sentences)
```

## Finetuning cohesive

cohesive users can finetune several parameters, which all impact the final segmentation results in different ways. Here is a quick summary:

- **Alpha**:

  - **Role**: Used as a weight in combining global and local similarities.
  - **Impact**: A higher alpha places more emphasis on global similarities, making the segmentation more influenced by overall similarity between sentences. Conversely, a lower alpha gives more weight to local similarities within the context window.
  - **Default**: 0.5

- **Context Window**:

  - **Role**: Determines the size of the context window used to calculate local similarities between sentences.
  - **Impact**: A smaller context window focuses on very close neighbors, capturing fine-grained local relationships. This may be suitable for documents where coherence is established within a small span of sentences. On the other hand, a larger context window considers a broader context, capturing longer-range dependencies and global patterns.
  - **Default**: 6

- **Decay**:

  - **Role**: Used to calculate decay factors based on distances between sentence indices when combining similarities.
  - **Impact**: A higher decay results in faster decay with increasing distance between sentences. This means that sentences further apart contribute less to the overall similarity, emphasizing local cohesion. A lower decay allows for longer-range dependencies to impact the segmentation.
  - **Default**: 0.8

- **Resolution**:

  - **Role**: Used in the community detection algorithm.
  - **Impact**: A higher resolution value leads to more and smaller communities, potentially yielding finer-grained segmentation. Conversely, a lower resolution results in fewer and larger communities, offering a more consolidated segmentation.
  - **Default**: 1.0

To modify the parameters, either pass in the appropriate parameter name and value when you call the create_segments method, or use the dedicated finetune_params function:

```python
# Via create_segments
cohesive.create_segments(sentences, context_window=3)

# Via finetune_params
cohesive.finetune_params(alpha=1, decay=0.2)
```

**Note: Any update to the parameters is stateful.**

At any time you can view the current parameters by calling the **get_params** method.

## Viewing the segments

When **create_segments** has finished, cohesive will print a summary of the total number of segments that were created.

To view the segments, simply call the **print_segments** method.

You can also view the start and end indices of sentences with a segment via the **print_segment_boundaries** function.

## References

cohesive is inspired by an article written by Massimiliano Costacurta, published in Towards Data Science in June 2023: [Text Tiling Done Right: Building Solid Foundations for your Personal LLM](https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1). The source code for this article can be accessed [here](https://github.com/massi82/texttiling/tree/master).
