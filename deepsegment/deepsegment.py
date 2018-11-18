from seqtag import predictor


class DeepSegment():
    seqtag_model = None
    def __init__(self, config_path):
        # loading the model
        DeepSegment.seqtag_model = predictor.load_model(config_path)
    
    def segment(self, text):
        if not DeepSegment.seqtag_model:
            print('Please load the model first')

        text = text.strip().split()
        tags = predictor.predict(DeepSegment.seqtag_model, text)
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