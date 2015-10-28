# DeepKnowledgeTracing
source code for the paper Deep Knowledge Tracing. http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf

At the moment the code doesn't include LSTM model, only RNN (I will upload LSTM when I get a chance).

Questions people have asked me:
----------
```
Q. How do you handle multiple students during training? It looks like sequences (ie. data of different 
students) of different length are padded to the same length.
A. Correct. This is very important for training speed.
```

```
Q. Does the training code have a termination condition?
A. No. I save a copy of the model each epoch and let training run until I feel like terminating it. You can 
start training 
from any saved model (where you left off).
```
