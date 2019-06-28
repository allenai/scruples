"""BERT baselines."""

from pytorch_pretrained_bert.modeling import (
    BertForMultipleChoice,
    BertForSequenceClassification)
from pytorch_pretrained_bert.tokenization import BertTokenizer
import skopt
import torch

from ..data.labels import Label
from ..dataset.transforms import (
    BertTransform,
    Compose,
    Map)


# the BERT sequence classification baseline

BERTClassifier = BertForSequenceClassification.from_pretrained
"""Predict fixed classes with a fine-tuned BERT model."""


BERT_CLASSIFIER_CONFIG = {
    'model': {
        # N.B. pretrained_model_name_or_path for the model must be the
        # same as pretrained_bert for the transform
        'pretrained_model_name_or_path': 'bert-large-uncased',
        'num_labels': len(Label)
    },
    'transform': {
        # N.B. pretrained_bert for the transform must be the same as
        # pretrained_model_name_or_path for the model
        'pretrained_bert': 'bert-large-uncased',
        'max_sequence_length': 512,
        'truncation_strategy_title': 'beginning',
        'truncation_strategy_text': 'beginning'
    }
}
"""Configuration for ``BERTClassifier``."""


BERT_CLASSIFIER_HYPER_PARAM_SPACE = [
    skopt.space.Real(
        low=1e-8,
        high=1e-1,
        prior='log-uniform',
        name='lr'),
    skopt.space.Real(
        low=1e-5,
        high=1e0,
        prior='log-uniform',
        name='weight_decay'),
    skopt.space.Real(
        low=0.0,
        high=0.6,
        prior='uniform',
        name='warmup_proportion'),
    skopt.space.Integer(
        low=1,
        high=50,
        name='n_epochs'),
    skopt.space.Integer(
        low=2,
        high=7,
        name='log_train_batch_size')
]
"""The hyper-param search space for ``BERTClassifier``."""


BERT_CLASSIFIER_TRANSFORM = (
    lambda
        pretrained_bert,
        max_sequence_length,
        truncation_strategy_title,
        truncation_strategy_text:
    Compose([
        BertTransform(
            tokenizer=BertTokenizer.from_pretrained(
                pretrained_bert,
                do_lower_case=pretrained_bert.endswith('-uncased')),
            max_sequence_length=max_sequence_length,
            truncation_strategy=(
                truncation_strategy_title,
                truncation_strategy_text
            )),
        lambda d: {
            'input_ids': torch.tensor(d['input_ids']),
            'attention_mask': torch.tensor(d['input_mask']),
            'token_type_ids': torch.tensor(d['segment_ids'])
        }
    ])
)
"""The factory to create data transforms for ``BERTClassifier``."""


# the BERT ranking baseline

BERTRanker = BertForMultipleChoice.from_pretrained
"""Rank choices with a softmax over a fine-tuned BERT model."""


BERT_RANKER_CONFIG = {
    'model': {
        # N.B. pretrained_model_name_or_path for the model must be the
        # same as pretrained_bert for the transform
        'pretrained_model_name_or_path': 'bert-large-uncased',
        'num_choices': 2
    },
    'transform': {
        # N.B. pretrained_bert for the transform must be the same as
        # pretrained_model_name_or_path for the model
        'pretrained_bert': 'bert-large-uncased',
        'max_sequence_length': 96
    }
}
"""Configuration for ``BERTRanker``."""


BERT_RANKER_HYPER_PARAM_SPACE = [
    skopt.space.Real(
        low=1e-8,
        high=1e-1,
        prior='log-uniform',
        name='lr'),
    skopt.space.Real(
        low=1e-5,
        high=1e0,
        prior='log-uniform',
        name='weight_decay'),
    skopt.space.Real(
        low=0.0,
        high=0.6,
        prior='uniform',
        name='warmup_proportion'),
    skopt.space.Integer(
        low=1,
        high=50,
        name='n_epochs'),
    skopt.space.Integer(
        low=2,
        high=7,
        name='log_train_batch_size')
]
"""The hyper-param seach space for ``BERTRanker``."""


BERT_RANKER_TRANSFORM = (
    lambda
        pretrained_bert,
        max_sequence_length:
    Compose([
        # wrap each action in a tuple for passing it to BertTransform
        lambda actions: tuple((action, None) for action in actions),
        # map BertTransform across all the action choices
        Map(
            transform=BertTransform(
                tokenizer=BertTokenizer.from_pretrained(
                    pretrained_bert,
                    do_lower_case=pretrained_bert.endswith('-uncased')),
                max_sequence_length=max_sequence_length,
                truncation_strategy=('beginning', 'beginning'))),
        # collect the action choices and stack their tensors so the
        # choices can be their own dimension of the batch
        lambda ds: {
            'input_ids': torch.stack([
                torch.tensor(d['input_ids'])
                for d in ds
            ], dim=0),
            'attention_mask': torch.stack([
                torch.tensor(d['input_mask'])
                for d in ds
            ], dim=0),
            'token_type_ids': torch.stack([
                torch.tensor(d['segment_ids'])
                for d in ds
            ], dim=0)
        }
    ])
)
"""The factory to create data transforms for ``BERTRanker``."""
