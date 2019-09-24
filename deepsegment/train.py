import random
random.seed(42)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import seqtag_keras

from seqtag_keras.utils import load_glove
from progressbar import progressbar

from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import re
import string

def bad_sentence_generator(sent, remove_punctuation = None):
    if not remove_punctuation:
        remove_punctuation = random.randint(0, 3)

    break_point = random.randint(1, len(sent)-2)
    lower_case = random.randint(0, 2)

    if remove_punctuation <= 1:
        # removing punctuation completely if remove_punctuation ==0 or ==1
        sent = re.sub('['+string.punctuation+']', '', sent)
    
    elif remove_punctuation == 2:
        # removing punctuation till a randomly selected point if remove_punctuation ==2
        if random.randint(0,1) == 0:
            sent = re.sub('['+string.punctuation+']', '', sent[:break_point]) + sent[break_point:]
        # removing punctuation after a randomly selected point if remove_punctuation ==2        
        else:
            sent = sent[:break_point] + re.sub('['+string.punctuation+']', '', sent[break_point:])    
    
    if lower_case <= 1:
        # lower casing sentence 
        sent = sent.lower()
    
    return sent

def generate_data(lines, max_sents_per_example=6, n_examples=1000):
    x, y = [], []
    
    for current_i in progressbar(range(n_examples)):
        x.append([])
        y.append([])

        chosen_lines = []
        for _ in range(random.randint(1, max_sents_per_example)):
            chosen_lines.append(random.choice(lines))
        
        chosen_lines = [bad_sentence_generator(line, remove_punctuation=random.randint(0, 3)) for line in chosen_lines]
        
        for line in chosen_lines:
            words = line.strip().split()
            for word_i, word in enumerate(words):
                x[-1].append(word)
                label = 'O'
                if word_i == 0:
                    label = 'B-sent'
                y[-1].append(label)
    
    return x, y

def train(x, y, vx, vy, epochs, batch_size, save_folder, glove_path):
    embeddings = load_glove(glove_path)
    
    checkpoint_path = os.path.join(save_folder, 'checkpoint')
    final_weights_path = os.path.join(save_folder, 'final_weights')
    params_path = os.path.join(save_folder, 'params')
    utils_path = os.path.join(save_folder, 'utils')    

    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, mode='max', monitor='f1')
    earlystop = EarlyStopping(patience=3, monitor='f1', mode='max')

    model = seqtag_keras.Sequence(embeddings=embeddings)
    
    model.fit(x, y, x_valid=vx, y_valid=vy, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, earlystop])

    model.save(final_weights_path, params_path, utils_path)
