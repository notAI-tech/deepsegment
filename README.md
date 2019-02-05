# Deep-Segmentation
A sentence segmenter that actually works!

The Demo is available at http://bpraneeth.com/projects

The code and pre-trained models for "DeepCorrection 1: Sentence Segmentation of unpunctuated text." as explained in the medium posts at https://medium.com/@praneethbedapudi/deepcorrection-1-sentence-segmentation-of-unpunctuated-text-a1dbc0db4e98 and
https://medium.com/@praneethbedapudi/deepsegment-2-0-multilingual-text-segmentation-with-vector-alignment-fd76ce62194f


The pre-trained models are available at https://github.com/bedapudi6788/DeepSegment-Models


# Requirements:
seqtag

```
# if you are using gpu for prediction, please see https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory for restricting memory usage

from deepsegment import DeepSegment
# the config file can be found at in the pre-trained model zip. Change the model paths in the config file before loading. 
# Since the complete glove embeddings are not needed for predictions, "glove_path" can be left empty in config file
segmenter = DeepSegment('path_to_config')
segmenter.segment('I am Batman i live in gotham')
['I am Batman', 'i live in gotham']
```

# To Do:
Add a sliding window for processing very long texts.

Update the seqtag model to work with tf 2.0 (Change to tf.data may be).

Train a single model for multi language segmentation.

# Notes:
Of all the sentence segmentation models I evaluated, without doubt deepsegment is the best in terms of accuracy in real word (bad punctuation, wrong punctuation)

I trained flair's ner model on the same data and flair has better results but, it's miniscule (0.3% absolute accuracy increase).

Since I want to keep using tf and keras for now, and since flair embeddings are not available for all the languages I want deepsegment to work on, I am going to keep using seqtag for this project.
