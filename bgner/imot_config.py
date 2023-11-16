# Model parameters

MODEL_PATH = 'bgner/model/imot_ner_model_2023-11-14'
TOKENIZER_PATH = 'bgner/model/imot_ner_tokenizer_2023-11-14'
# MODEL_PATH = 'bert-base-multilingual-cased'
# TOKENIZER_PATH = 'bert-base-multilingual-cased'
ENCODING = 'windows-1251'

TAG_TO_ID = {
    'O': 0,
    'B-FAC': 1,
    'I-FAC': 2
}

TRAINING_PARAMS = {
    'base_model': 'bert-base-multilingual-cased',
    'max_len': 128,
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 1e-5,
    'test_sample_size': 0.2,
    'val_sample_size': 0.5,  # From test_sample_size
    'random_state': 171
}
