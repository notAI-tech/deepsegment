# DeepSegment: A sentence segmenter that actually works!
Note: For the original implementation please use the "master" branch of this repo.

The Demo for deepsegment (en) + deeppunct is available at http://bpraneeth.com/projects/deeppunct

# Installation:
```
pip install --upgrade deepsegment
```

# Supported languages:
en - english (Trained on data from various sources)

fr - french (Only Tatoeba data)

it - italian (Only Tatoeba data)


# Usage:

```
from deepsegment import DeepSegment
# The default language is 'en'
segmenter = DeepSegment('en')
segmenter.segment('I am Batman i live in gotham')
# ['I am Batman', 'i live in gotham']

```

# Using with tf serving docker image
```
docker pull bedapudi6788/deepsegment_en:v2
docker run -d -p 8500:8500 bedapudi6788/deepsegment_en:v2
```

```
from deepsegment import DeepSegment
# The default language is 'en'
segmenter = DeepSegment('en', tf_serving=True)
segmenter.segment('I am Batman i live in gotham')
# ['I am Batman', 'i live in gotham']
```

Training deepsegment on custom data: https://colab.research.google.com/drive/1CjYbdbDHX1UmIyvn7nDW2ClQPnnNeA_m
