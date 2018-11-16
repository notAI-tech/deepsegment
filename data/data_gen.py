import re
import string
import pickle
import random
random.seed(6788)
import progressbar

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


def data_gen(all_sents, max_no_sents = 10, data_len = 10000, file_name = 'data.txt', remove_punctuation = None):
    print('Generating the file', file_name, max_no_sents)

    f = open(file_name, 'w')
    for _ in progressbar.progressbar(range(data_len)):
        sents = []
        for __ in range(random.randint(0, max_no_sents)):
            sents.append(bad_sentence_generator(random.choice(all_sents), remove_punctuation=remove_punctuation))
        for sent in sents:
            for i, word in enumerate(sent.split()):
                if i == 0:
                    label = 'B-sent'
                else:
                    label = 'O'
                
                f.write(word.strip() + '\t' + label + '\n')
        
        f.write('\n')






if __name__ == '__main__':
    all_sents = open('golden_english_sentences.txt').readlines()
    all_sents = [i.strip() for i in all_sents]
    # for i in range(10):
    #     sent = random.choice(all_sents)
    #     print(sent, bad_sentence_generator(sent))
    data_gen(all_sents[150000:], data_len=1000000, file_name='train.txt')
    data_gen(all_sents[20000:150000], data_len=100000, file_name='valid.txt')
    #For generating correctly punctuated data but might be wrong cased
    data_gen(all_sents[:20000], data_len=1000, file_name='test_correct_punct.txt', remove_punctuation=10)
    #For generating punctuation removed data, might be wrong cased
    data_gen(all_sents[:20000], data_len=1000, file_name='test_no_punct.txt', remove_punctuation=1)
    #For generating partial punctuation removed data, might be wrong cased
    data_gen(all_sents[:20000], data_len=1000, file_name='test_partial_punct.txt', remove_punctuation=2)
