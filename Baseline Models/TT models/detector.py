import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import argparse
import pandas as pd
from datasets import load_datasets

parser = argparse.ArgumentParser(description='GPT-2 Output Detector')
parser.add_argument('--name', required=True, type=str, help='[gpt1, transfo_xl]', default='gpt2')
opt = parser.parse_args()

def load_data(name = opt.name):
    TT = load_dataset('turingbench / TuringBench', name = f'TT_{name}', split ='test')
    data = pd.DataFrame.from_dict(TT)
    
    return data

data = load_data(opt.name)


class Detector(object):

    def __init__(self):

        print('Initializing Detector...')

        data = torch.load('/home/azu5030/TB/baseline/detector/detector-large.pt')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-large')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval().cuda()

    def predict(self, txt):

        tokens = self.tokenizer.encode(txt, max_length=self.tokenizer.max_len)
        #tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        tokens = torch.Tensor(tokens)

        tokens = tokens.unsqueeze(0).cuda().long()

        mask = torch.ones_like(tokens).cuda().long()

        logits = self.model(tokens, attention_mask=mask)

        probs = logits[0].softmax(dim=-1)

        probs = probs.detach().cpu().flatten().numpy()

        return probs


    def get_result(self, txt):
        p = self.predict(txt)
        prob = np.max(p)
        result = np.argmax(p)
        if result == 1:
            result = 'human'
        else:
            result = 'machine'
        print(result, ' | ', prob)



from sklearn.metrics import classification_report as cr

if __name__ == '__main__':

    text = 'GPT-2 detector'

    #det = Detector()
    #p = det.predict(text)
    #prob = np.max(p)
    #result = np.argmax(p)

    
    #if result == 1:
     #   result = 'Human'
    #else:
     #   result = 'Machine'
    #print(result, ' | ', prob)


    def change_label(data):
        for i in range(len(data)):
            if data[i] == 'human':
               data[i] = 1
            else:
                data[i] = 0
        return data
    

    def det(data):

        print('Running: ', opt.name)
        det = Detector()

        pred = []
        
        for i in list(data['Generation']):
            p = det.predict(str(i))
            prob = np.max(p)
            #print(prob)
            result = np.argmax(p)
            #print('rounded prob: ', result)
            
            if result == 1:
                result = 'human'
            else:
                result = 'machine'

            pred.append(result)

        y_true = change_label(list(data['label']))
        y_pred = change_label(pred)
        print(cr(y_true=y_true, y_pred=y_pred, digits=4))

    DATA = data
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    det(data = DATA)
