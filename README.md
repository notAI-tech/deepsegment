# DeepSegment: A sentence segmenter that actually works!
Note: For the original implementation please use the "master" branch of this repo.

The Demo for deepsegment (en) + deeppunct is available at http://bpraneeth.com/projects/deeppunct

# Installation:
```bash
pip install --upgrade deepsegment
```

# Supported languages:
en - english (Trained on data from various sources)

fr - french (Only Tatoeba data)

it - italian (Only Tatoeba data)


# Usage:

```python
from deepsegment import DeepSegment
# The default language is 'en'
segmenter = DeepSegment('en')
segmenter.segment('I am Batman i live in gotham')
# ['I am Batman', 'i live in gotham']

```

# Using with tf serving docker image
```bash
docker pull bedapudi6788/deepsegment_en:v2
docker run -d -p 8500:8500 bedapudi6788/deepsegment_en:v2
```

```python
from deepsegment import DeepSegment
# The default language is 'en'
segmenter = DeepSegment('en', tf_serving=True)
segmenter.segment('I am Batman i live in gotham')
# ['I am Batman', 'i live in gotham']
```

# Finetuning DeepSegment
Since one-size will never fit all, finetuning deepsegment's default models with your own data is encouraged.

```python
from deepsegment import finetune, generate_data

x, y = generate_data(['my name', 'is batman', 'who are', 'you'], n_examples=10000)
vx, vy = generate_data(['my name', 'is batman'])

# NOTE: name, epochs, batch_size, lr are optional arguments.
finetune('en', x, y, vx, vy, name='finetuned_model_name', epochs=number_of_epochs, batch_size=batch_size, lr=learning_rate)
```

# Using with a finetuned checkpoint
```python
from deepsegment import DeepSegment
segmenter = DeepSegment('en', checkpoint_name='finetuned_model_name')
```


Training deepsegment on custom data: https://colab.research.google.com/drive/1CjYbdbDHX1UmIyvn7nDW2ClQPnnNeA_m
