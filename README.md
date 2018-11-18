# Deep-Segmentation
Sentence Segmentation of un-punctuated text.

Place holder for the code and pre-trained models for "DeepCorrection 1: Sentence Segmentation of unpunctuated text." as explained in the medium post at https://medium.com/@praneethbedapudi/deepcorrection-1-sentence-segmentation-of-unpunctuated-text-a1dbc0db4e98 .


The pre-trained models is available at https://drive.google.com/open?id=1keUOKjloauUvhAhxErPMZjjkfA2tPnXH

The data is available at https://drive.google.com/open?id=1inDBFHZA8pKhVdFB-I4Vkk3tEuxzt6Dv


# Requirements:
seqtag

```
from seqtag import predictor
from deepsegment import segment
# the config file can be found at in the pre-trained model zip. Change the model paths in the config file before loading. 
# Since the complete glove embeddings are not needed for predictions, "glove_path" can be left empty in config file
seqtag_model = predictor.load_model(path_to_config_file)
segment('I am Batman i live in gotham', seqtag_model)
['I am Batman', 'i live in gotham']
```
