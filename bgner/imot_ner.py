import re
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from bgner import imot_config as conf


class ImotNer:
    """
    Bulgarian NER trained specifically for recognizing streets/locations in Sofia city center
    Main functionalities:
        - Finetune of existing Bert model (base or already fine-tuned for NER)
        - Detailed Model evaluation functionalities
        - Export predictions in format for training that always manual correction and continues fine-tuning
        - Inference
    """

    def __init__(self, mode='predict', device=None):

        if not device:  # Allows to force the usage of CPU
            if mode == 'train':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.tp = conf.TRAINING_PARAMS

        self.tag_to_id = conf.TAG_TO_ID
        self.id_to_tag = {self.tag_to_id[tag]: tag for tag in self.tag_to_id}

        self.proc_date = datetime.now().strftime('%Y-%m-%d')

        self.model = AutoModelForTokenClassification.from_pretrained(conf.MODEL_PATH, num_labels=len(self.tag_to_id))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.tp['learning_rate'])
        self.tokenizer = AutoTokenizer.from_pretrained(conf.TOKENIZER_PATH)

    class NERDataset(Dataset):
        """
        Dataset class to be passed in the DataLoader
        """

        def __init__(self, data, device):
            super().__init__()
            self.data = data
            self.device = device

        def __getitem__(self, index):
            item = self.data[index]
            item['input_ids'] = item['input_ids'].to(self.device)
            item['attention_mask'] = item['attention_mask'].to(self.device)
            item['labels'] = item['labels'].type(torch.LongTensor)
            item['labels'] = item['labels'].to(self.device)
            return item

        def __len__(self):
            return len(self.data)

    def split_sentences(self, s: str) -> list:
        """
        Applies different regex patterns to replace '.' with [DOTREP] where the '.' is not an end of a sentence.
        Then splits the string into sentences and adss \n at the end. The output can be then used for tokenization

        :param s: String with the complete description
        :return: List with the parsed sentences
        """

        def dot_replacements(x: str, char_list: list, pattern: str, replacement: str) -> str:
            """
            Helper Function: Dots are pain in the ass when it comes to sentence splits. They can be anywhere
            within the sentence as part of abbreviations. This is a weak attempt to coop with this.

            :param x: a string with the text that would be processed
            :param char_list: list of all characters that could be followed by a dot that is not the end of a sentence
            :param pattern: the regex pattern as an f string with a placeholder for the char
            :param replacement: f string with the replacement
            :return: modified string
            """

            for char in char_list:
                p = rf'{char}{pattern}'
                r = rf'{char}{replacement}'
                x = re.sub(p, r, x)

            return x

        chars_1 = ['М', 'м', 'Кв', 'кв', 'Ел', 'ел', 'Ап', 'ап', 'Ет', 'ет']
        chars_2 = ['Ул', 'ул', 'Бул', 'бул', 'Гр', 'гр', 'Пл', 'пл', 'Реф', 'реф']
        chars_3 = ['Кв', 'кв']
        chars_4 = ['.', ',', '!', '?', '...', ':', '-', ';', '[DOTREP]']

        s1 = dot_replacements(s, chars_1, r'\.(\s*[а-яьъ0-9,\.])',
                              r"[DOTREP]\1")  # Words that might be at the end of a sentence
        s1 = dot_replacements(s1, chars_2, r'\.', r'[DOTREP]')  # Words that are unlikely to be at the end of a sentence
        s1 = dot_replacements(s1, [''], r'(\d+)\.(\d+)', r'\1[DOTREP]\2')  # Decimal numbers
        s1 = dot_replacements(s1, [''], r'(\s[А-Я]{1})\.', r'\1[DOTREP]')  # Name abbreviations numbers
        s1 = dot_replacements(s1, chars_3, r'\.(\s*[Център,\.])', r"[DOTREP]\1")  # Frequent special case

        for char in chars_4:
            s1 = s1.replace(char, f' {char} ')
        s1 = re.sub(r'\s+', ' ', s1)

        sentences = re.split(r'(?<=[.!?...])\s*', s1)
        sentences = [_.replace('[DOTREP]', '.') for _ in sentences]
        sentences = [_ + '\n' for _ in sentences]

        return sentences

    def remove_chars(self, input_string: str, chars_to_remove: str) -> str:
        """
        Removes residual charcters from a string - e.g. quotes
        """

        for char in chars_to_remove:
            input_string = input_string.replace(char, '')

        return input_string

    def combine_subtokens(self, tokens_with_labels) -> list:
        """
        Aggregates the tokenized sub-words into whole words and selects the label of the first token for the word

        :param tokens_with_labels:
        :return: List of lists with the whole words and the labels
        """
        combined_text = ""
        for token, _ in tokens_with_labels:
            if token.startswith("##"):
                combined_text = combined_text + token[2:]  # Remove the ## and append to the previous word
            else:
                combined_text += " " + token

        combined_text = self.remove_chars(combined_text, ["'", '"'])
        combined_text = combined_text.strip()
        word_tokens = combined_text.split()

        # Ugly way to get the tokens - it is even uglier as it is possible to get sub-token to be flagged as B-FAC
        # even if the first token of the word is marked as O.
        word_labels = [_[1] for _ in tokens_with_labels if
                       _[0] == tokens_with_labels[0][0] or not _[0].startswith('##')]

        return [word_tokens, word_labels]

    def predict_for_training(self, input_text: str) -> pd.DataFrame:
        """
        Uses existing model to create a labeled dataset that can be
        manually reviewed and used for fine-tuning/retraining.

        :param input_text: a string with the full ad description
        :return: a string with the predicted Named Entities (NOT coma separated)
        """

        self.model.eval()
        # self.model.to('cpu')
        training_result = pd.DataFrame()
        sentences = self.split_sentences(input_text)

        for sent_id, sentence in enumerate(sentences):
            entities = []
            tokens = self.tokenizer(sentence, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**tokens)

            predicted_labels = torch.argmax(outputs.logits, dim=2)[0].tolist()[1:-1]
            tokenized_input = self.tokenizer.tokenize(sentence)

            for token, label in zip(tokenized_input, predicted_labels):
                entities.append((token, label))

            whole_tokens = imot.combine_subtokens(entities)
            training_result = pd.concat([training_result, pd.DataFrame({'sentence_order': sent_id,
                                                                        'word': whole_tokens[0],
                                                                        'tag': whole_tokens[1]})])
            training_result['word_order'] = training_result.groupby('sentence_order')['word'].cumcount()

        return training_result

    def predict(self, input_text: str) -> str:
        """
        Predicts Named Entities.

        :param input_text: a string with the full ad description
        :return: a string with the predicted Named Entities (NOT coma separated)
        """
        self.model.eval()

        entities = []
        sentences = self.split_sentences(input_text)

        for sentence in sentences:
            tokens = self.tokenizer(sentence, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**tokens)

            predicted_labels = torch.argmax(outputs.logits, dim=2)[0].tolist()[1:-1]
            tokenized_input = self.tokenizer.tokenize(sentence)

            for token, label in zip(tokenized_input, predicted_labels):
                # print(token, label)
                if label > 0:
                    entities.append((token, label))

        predictions = self.combine_subtokens(entities)

        # Clean up and format the predictions
        if len(predictions[0]) < 1:
            return ''

        elif len(predictions[0]) == 1:
            return predictions[0][0]

        named_entities = predictions[0][0]
        for i in range(1, len(predictions[0])):
            if predictions[1][i] == 1:
                named_entities = named_entities + ', ' + predictions[0][i]
            else:
                named_entities = named_entities + ' ' + predictions[0][i]

        named_entities = list(set([_.replace(" .", ".").strip() for _ in named_entities.split(',')]))
        named_entities = ', '.join(named_entities)

        return named_entities

    def split_dataset(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Splits a pre-processed dataset into training, validation and testing.

        :param df: DataFrame with every token as a row and indexes for the sentence and word order
        :return: raw tokens and labels list of each dataset group
        """

        train_ids, rest_ids = train_test_split(df['ad'].unique(),
                                               test_size=self.tp['test_sample_size'],
                                               random_state=self.tp['random_state'])

        val_ids, test_ids = train_test_split(rest_ids,
                                             test_size=self.tp['val_sample_size'],
                                             random_state=self.tp['random_state'])

        datasets = {}
        datasets['train_desc'] = [_ for _ in
                                  df.query("ad.isin(@train_ids)").groupby(['ad', 'sentence_order'])['word'].apply(list)]
        datasets['train_labels'] = [_ for _ in
                                    df.query("ad.isin(@train_ids)").groupby(['ad', 'sentence_order'])['tag'].apply(
                                        list)]
        datasets['val_desc'] = [_ for _ in
                                df.query("ad.isin(@val_ids)").groupby(['ad', 'sentence_order'])['word'].apply(list)]
        datasets['val_labels'] = [_ for _ in
                                  df.query("ad.isin(@val_ids)").groupby(['ad', 'sentence_order'])['tag'].apply(list)]
        datasets['test_desc'] = [_ for _ in
                                 df.query("ad.isin(@test_ids)").groupby(['ad', 'sentence_order'])['word'].apply(list)]
        datasets['test_labels'] = [_ for _ in
                                   df.query("ad.isin(@test_ids)").groupby(['ad', 'sentence_order'])['tag'].apply(list)]

        self.train_ids = train_ids,
        self.val_ids = val_ids,
        self.test_ids = test_ids

        return datasets

    def adjust_labels(self, tokenized_input, tokenized_labels) -> list:
        """
        Aligns the labels with the sub-words tokens by propagating the label of the first sub-token

        :param tokenized_input:
        :param tokenized_labels:
        :return:
        """
        all_adjusted_labels = []
        data = []

        print(len(tokenized_input["input_ids"]))

        k = 0
        for k in range(0, len(tokenized_input["input_ids"])):  # Loop over all sentences
            print(k)
            prev_wid = -1
            word_ids_list = tokenized_input.word_ids(batch_index=k)
            existing_label_ids = [self.tag_to_id[_] for _ in tokenized_labels[k]]

            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:  # Loop over all words/tokens
                if wid is None:
                    adjusted_label_ids.append(-100)
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])

            res = {'input_ids': tokenized_input["input_ids"][k],
                   'attention_mask': tokenized_input["attention_mask"][k],
                   'labels': torch.tensor(adjusted_label_ids)}

            data.append(res)

        all_adjusted_labels.append(adjusted_label_ids)

        return data

    def train_tokenize(self, dataset, labels):
        """
        Tokenizes for training...

        :param dataset: PyTorch tensor
        :param labels: PyTorch tensor
        :return: PyTorch dataset
        """

        inputs = self.tokenizer.batch_encode_plus(dataset,
                                                  padding='max_length',
                                                  max_length=self.tp['max_len'],
                                                  truncation=True,
                                                  is_split_into_words=True,
                                                  return_tensors="pt")

        inputs_with_labels = self.adjust_labels(inputs, labels)

        return self.NERDataset(inputs_with_labels, self.device)

    def print_test(self, loader, stage: str, dataset: str) -> list:
        """
        Prints Accuracy, precision and recall by class before and after the training. Also exports raw sub-tokens
        with predictions and labels for detailed review

        :param loader: DataLoader object - train, val or test
        :param stage: Pre-Training, Epoch <E>, Post-Training
        :param dataset: Training, Validation or Test
        :return: list with all the metrics and detailed df on sub-tokens for detailed review
        """

        results = self.test(loader)

        accuracy = round(results[0], 4)
        precision = [round(_, 4) for _ in results[2]]
        recall = [round(_, 4) for _ in results[3]]
        print(f'| {stage} | {dataset} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}')

        return results

    def data_loader(self, df: pd.DataFrame) -> None:
        """

        :param df: Loaded csv with a proper format - every token (space split) is a row with a label
        :return: Creates the loader datasets
        """

        ds = self.split_dataset(df)
        train_dataset = self.train_tokenize(ds['train_desc'], ds['train_labels'])
        val_dataset = self.train_tokenize(ds['val_desc'], ds['val_labels'])
        test_dataset = self.train_tokenize(ds['test_desc'], ds['test_labels'])

        self.train_loader = DataLoader(train_dataset, batch_size=self.tp['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.tp['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.tp['batch_size'], shuffle=True)

    def train_pipeline(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        Executes the training and while doing it, prints nice looking metrics

        :param df: Loaded csv with a proper format - every token (space split) is a row with a label
        :param save_path: Optional - if provided the model would be saved in this path
        :return:
        """

        self.data_loader(df)
        self.print_test(self.val_loader, 'Pre-Training', 'Validation')

        for epoch in range(1, self.tp['epochs'] + 1):
            self.train(epoch)
            self.print_test(self.val_loader, f'Epoch {epoch}', 'Validation')

        self.model_metrics = self.print_test(self.test_loader, 'Post-Training', 'Test')
        eval_df = self.model_metrics[4]

        if save_path:
            self.model.save_pretrained(f'{save_path}/imot_ner_model_{self.proc_date}')
            self.tokenizer.save_pretrained(f'{save_path}/imot_ner_tokenizer_{self.proc_date}')

            eval_dir = f'{save_path}/imot_ner_eval_{self.proc_date}'
            if not os.path.exists(eval_dir):
                os.mkdir(eval_dir)

            eval_df.to_csv(f'{save_path}/imot_ner_eval_{self.proc_date}/test_review.csv', index=False)
            model_summary = {'train_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             'base_model': self.tp['base_model'],
                             'classes': ', '.join(self.tag_to_id.keys()),
                             'data_size': {'train_set_size': len(self.train_loader.dataset),
                                           'val_set_size': len(self.val_loader.dataset),
                                           'test_set_size': len(self.test_loader.dataset)},
                             'performance': {'accuracy': round(self.model_metrics[0], 4),
                                             'precision': ', '.join(str(round(_, 4)) for _ in self.model_metrics[2]),
                                             'recall': ', '.join(str(round(_, 4)) for _ in self.model_metrics[3])}
                             }
            with open(f'{save_path}/imot_ner_eval_{self.proc_date}/model_summary.json', 'w') as json_file:
                json.dump(model_summary, json_file, indent=4)

    def train(self, epoch):
        """
        Trains the model...

        :param epoch: How many EPOCHs - it is coming from the config
        :return:
        """

        self.model.train()
        total_loss = 0
        log_interval = round(len(self.train_loader) / 10)

        for batch_index, batch in enumerate(self.train_loader):
            self.model.zero_grad()
            output = self.model(**batch)
            loss = output[0]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_index % log_interval == 0 and batch_index > 0:
                current_loss = total_loss / log_interval
                print('| epoch {:3d} | '
                      '{:5d}/{:5d} batches | '
                      'loss {:5.2f}'.format(
                    epoch,
                    batch_index, len(self.train_loader),
                    current_loss))
                total_loss = 0

    def test(self, data_loader) -> list:
        """
        Generates precision, recall, confusion matrix and sub-token level df with predictions and labels

        :param data_loader:
        :return: list with all the metrics
        """
        self.model.eval()
        total_score = 0
        total_len = 0
        all_labels = []
        all_preds = []
        all_input_ids = []

        with torch.no_grad():
            for batch_index, batch in enumerate(data_loader):
                output = self.model(**batch)
                preds = np.argmax(output[1].cpu(), axis=2)
                preds = preds.to(self.device)
                preds = preds[(batch['labels'] != -100)]
                labels = batch['labels'][(batch['labels'] != -100)]
                total_score += preds.eq(labels.cpu().to(self.device)).sum()
                total_len += len(labels)
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
                all_input_ids.extend(batch['input_ids'][(batch['labels'] != -100)].tolist())

        eval_df = pd.DataFrame({'input_id': all_input_ids,
                                'token': self.tokenizer.convert_ids_to_tokens(all_input_ids),
                                'label': all_labels,
                                'prediction': all_preds,
                                'match': [1 if label == pred else 0 for label, pred in zip(all_labels, all_preds)]})

        conf_matrix = confusion_matrix(eval_df['label'], eval_df['prediction'], normalize='true')
        precision = precision_score(eval_df['label'], eval_df['prediction'], average=None)
        recall = recall_score(eval_df['label'], eval_df['prediction'], average=None)

        return [(total_score.item() / total_len), conf_matrix, precision, recall, eval_df]


if __name__ == '__main__':

    # Don't forget to load some data first!
    ads_descriptions = pd.read_csv('data/ads_latest_202311091517.csv')
    ads_tokens_for_training = pd.read_csv('data/imot_labeled_data_word_split_20231110.csv', sep='\t')

    i = 0
    r = ads_descriptions.shape[0]

    # Predicting for training
    imot = ImotNer(mode='predict')
    ads_descriptions['locations'] = ''

    training_result = pd.DataFrame()

    for i in range(r):
        print(i)
        s = ads_descriptions.iloc[i]['ad_description']
        predictions = imot.predict_for_training(s)
        predictions['ad'] = ads_descriptions.iloc[i]['ad_url'].split("=")[-1]
        training_result = pd.concat([training_result, predictions])

    training_result = training_result[['ad', 'sentence_order', 'word_order', 'word', 'tag']]
    training_result['tag'] = training_result['tag'].apply(lambda x: imot.id_to_tag[int(x)])
    training_result.to_csv('data/imot_labeled_data_word_split_20231110.csv', index=False)

    # Training
    imot = ImotNer(mode='train')
    imot.train_pipeline(df=ads_tokens_for_training, save_path='model')

    # Evaluating existing model (no need to run train first)
    imot = ImotNer(mode='train')
    imot.data_loader(ads_tokens_for_training)
    eval_df = imot.print_test(imot.val_loader, 'Manual Evaluation', 'Evaluation')
    eval_df.to_csv('model\imot_ner_eval_2023-11-10/manual_eval.csv', index=False)

    # -- Review predictions on the test set
    ads_descriptions['ad'] = ads_descriptions['ad_url'].apply(lambda x: x.split("=")[-1]).astype(str)
    test_ads_descriptions = ads_descriptions[ads_descriptions['ad'].isin(imot.test_ids)]
    test_ads_descriptions['locations'] = ''

    imot = ImotNer(mode='predict')

    res = []
    for i in range(test_ads_descriptions.shape[0]):
        print(i)
        s = test_ads_descriptions.iloc[i]['ad_description']
        res.append(imot.predict(s))

    test_ads_descriptions.loc[0:r, 'locations'] = res
    empty_location = test_ads_descriptions['ad_description'].loc[276]

    # Predicting for inference
    imot = ImotNer(mode='predict')
    ads_descriptions['locations'] = ''
    res = []

    for i in range(r):
        print(i)
        s = ads_descriptions.iloc[i]['ad_description']
        res.append(imot.predict(s))

    ads_descriptions['locations'].iloc[0:r] = res
