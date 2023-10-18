import pandas as pd
from deeppavlov import build_model
from deeppavlov import evaluate_model
import csv

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

class NERTrainer:

    def __init__(self, data, seed=17):
        self.data = pd.read_csv(data)
        self.seed = seed

    def create_datasets(self, split):


corpus = pd.read_csv('ner/ads_latest_202310091115.csv')
ad_list = corpus['ad_url'].unique()
train_ids, test_ids = train_test_split(ad_list, test_size=0.2, random_state=42)

labeled_tokens = pd.read_csv('ner/imot_data.csv', sep='\t', encoding='windows-1251', quoting=csv.QUOTE_NONE)
train_tokens = labeled_tokens[labeled_tokens['ad'].isin(train_ids)]
test_tokens = labeled_tokens[labeled_tokens['ad'].isin(test_ids)]

train_corpus = corpus[corpus['ad_url'].isin(train_ids)]
test_corpus = corpus[corpus['ad_url'].isin(test_ids)]

ner_model = build_model('ner_ontonotes_bert_mult', download=True, install=True)

test_descriptions = [_ for _ in test_corpus['ad_description']]
test_urls = [_ for _ in test_corpus['ad_url']]

res = pd.DataFrame()
i = 0
for i in range(len(test_ids)):
    print(i)

    x = ner_model([test_descriptions[i][0:512]])
    df = pd.DataFrame({'ad': test_urls[i], 'word': x[0][0], 'tag': x[1][0]})

    res = pd.concat([res, df])







eval_model = evaluate_model('ner_ontonotes_bert_mult', download=True)

ner_model = build_model('ner_ontonotes_bert_mult', download=True, install=True)

ads = pd.read_csv("ner/ads_latest_202310091115.csv")

res = pd.DataFrame()

ads['desc'] = ['Продава апартамент '] * ads.shape[0] + pd.Series(np.where(ads['ad_street'] == 'Център', 'в ', 'на ')) + ads['ad_street'] + '. ' + ads['ad_description']

descriptions = [_ for _ in ads['desc']]
ad_urls = [_.split("=")[-1] for _ in ads['ad_url']]

i = 0
for i in range(len(ad_urls)):
    print(i)

    x = ner_model([descriptions[i][0:512]])
    df = pd.DataFrame({'ad': ad_urls[i], 'word': x[0][0], 'tag': x[1][0]})

    res = pd.concat([res, df])

res.to_csv('ner/imot_data_set_baseline.csv', index=False)