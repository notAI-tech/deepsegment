import nltk
import spacy
from time import time
from seqtag import predictor

seqtag_model = predictor.load_model('/Users/bedapudi/Downloads/deepsegment_data/example_config.json')

spacy_nlp = spacy.load('en')

correct_punct_lines = open('data/test_correct_punct.txt').read().split('\n\n')
no_punct_lines = open('data/test_no_punct.txt').read().split('\n\n')
partial_punct_lines = open('data/test_partial_punct.txt').read().split('\n\n')

correct_punct_lines = [[[word_tag.split()[0] for word_tag in line.strip().split('\n')], [word_tag.split()[1] for word_tag in line.strip().split('\n')]] for line in correct_punct_lines if line.strip()]
no_punct_lines = [[[word_tag.split()[0] for word_tag in line.strip().split('\n')], [word_tag.split()[1] for word_tag in line.strip().split('\n')]] for line in no_punct_lines if line.strip()]
partial_punct_lines = [[[word_tag.split()[0] for word_tag in line.strip().split('\n')], [word_tag.split()[1] for word_tag in line.strip().split('\n')]] for line in partial_punct_lines if line.strip()]

def preprocess(text):
    text = text.strip()
    text = text.split()
    return text

def segment(text, tags):
    text = preprocess(text)
    sents = []
    
    current_sent = []
    for i, word in enumerate(text):
        if tags[i] == 'B-sent':
            if current_sent:
                sents.append(' '.join(current_sent))
            current_sent = [word]
        else:
            current_sent.append(word)
    
    sents.append(' '.join(current_sent))
    
    return sents


def get_deep_segment_accuracy():
    for lines in [correct_punct_lines, no_punct_lines, partial_punct_lines]:
        count = 0
        for words, tags in lines:
            predicted_tags = predictor.predict(seqtag_model, words)
            if predicted_tags == tags:
                count = count + 1

        print('Total:', count, len(lines), (100*count)/len(lines))

def get_nltk_accuracy():
    for lines in [correct_punct_lines, no_punct_lines, partial_punct_lines]:
        count = 0
        for words, tags in lines:
            predicted_sentences = nltk.sent_tokenize(' '.join(words))
            correct_sentences = segment(' '.join(words), tags)
            if predicted_sentences == correct_sentences:
                count = count + 1

        print('Total:', count, len(lines), (100*count)/len(lines) )

def get_spacy_accuracy():
    for lines in [correct_punct_lines, no_punct_lines, partial_punct_lines]:
        count = 0
        for words, tags in lines:
            predicted_sentences = [sent.text for sent in spacy_nlp(' '.join(words)).sents]
            correct_sentences = segment(' '.join(words), tags)
            if predicted_sentences == correct_sentences:
                count = count + 1

        print('Total:', count, len(lines), (100*count)/len(lines))


print('--------------- Deep Segment')
start_time = time()
get_deep_segment_accuracy()
print('Deep Segment Time:', time() - start_time)
start_time = time()

print('--------------- NLTK')
get_nltk_accuracy()
print('NLTK Time:', time() - start_time)
start_time = time()

print('--------------- Spacy')
get_spacy_accuracy()
print('Spacy Time:', time() - start_time)
start_time = time()