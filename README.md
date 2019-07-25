# sequence-labeling-tutorial
In machine learning, sequence labeling is a type of pattern recognition task that involves the algorithmic assignment of a categorical label to each member of a sequence of observed values.  An important example is Named-entity recognition (NER). We take NER for example in this tutorial.

Named-entity recognition (NER) is a sub task within information extraction, that seeks to recognize and classify named entity mentions in text into pre-defined semantic categories((e.g., location, organization, geo-political entity, person). For example, given a sentence "Cook bought two hundred shares of Apple Inc. in 2003.", the name entities can be highlighted as:

$[Cook]_{Person} \ bought \ two \ hundred \ shares \ of \ [Apple\  Inc.]_{Organization} \ in  \ [2003]_{Time}.$

In this example, a one-token person name, a two-token company name and a temporal expression have been located and classified.

NER has a wide range of applications in the industry. It can be used in recognizing relevant entities in customer complaints, which will be classified accordingly and forwarded to the appropriate department responsible for the identified product. It can also be used in search engine algorithms, to extract entities and compare them with the tags associated with the website articles for a quick and efficient search.

NER problem can be viewed as a sequence labeling problem, a simple class of structural prediction problems. The goal is to assign one discrete label to each member of a sequence of input tokens. We may have hearde of Hidden Markov Model (HMM), Maximum Entropy Markov Models (MEMM) and Conditional Random Fields (CRF), these models can be applied on sequence labeling tasks.

In this tutorial, we will implement a sequence labeling system using CRF, then evaluate it on a NER task. The model can also be extended to other sequence labeling tasks like part of speech tagging, chunking, etc. The major differences lie in the datasets and metrics.
