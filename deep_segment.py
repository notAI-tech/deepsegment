from seqtag import predictor


def preprocess(text):
    text = text.strip()
    text = text.split()
    return text


def segment(text, seqtag_model):
    text = preprocess(text)
    tags = predictor.predict(seqtag_model, text)
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
