"""RoBERTa baselines."""

from transformers import (
    RobertaForMultipleChoice,
    RobertaForSequenceClassification,
    RobertaTokenizer)
import skopt
import torch

from ..data.labels import Label
from ..dataset.transforms import (
    BertTransform,
    Compose,
    Map)


# the RoBERTa sequence classification baseline

RoBERTaClassifier = RobertaForSequenceClassification.from_pretrained
"""Predict fixed classes with a fine-tuned RoBERTa model."""


ROBERTA_CLASSIFIER_CONFIG = {
    'model': {
        # N.B. pretrained_model_name_or_path for the model must be the
        # same as pretrained_roberta for the transform
        'pretrained_model_name_or_path': 'roberta-large',
        'num_labels': len(Label)
    },
    'transform': {
        # N.B. pretrained_roberta for the transform must be the same as
        # pretrained_model_name_or_path for the model
        'pretrained_roberta': 'roberta-large',
        'max_sequence_length': 512,
        'truncation_strategy_title': 'beginning',
        'truncation_strategy_text': 'beginning'
    }
}
"""Configuration for ``RoBERTaClassifier``."""


ROBERTA_CLASSIFIER_HYPER_PARAM_SPACE = [
    skopt.space.Real(
        low=1e-8,
        high=1e-2,
        prior='log-uniform',
        name='lr'),
    skopt.space.Real(
        low=1e-5,
        high=1e0,
        prior='log-uniform',
        name='weight_decay'),
    skopt.space.Real(
        low=0.0,
        high=1.0,
        prior='uniform',
        name='warmup_proportion'),
    skopt.space.Integer(
        low=1,
        high=10,
        name='n_epochs'),
    skopt.space.Integer(
        low=3,
        high=10,
        name='log_train_batch_size')
]
"""The hyper-param search space for ``RoBERTaClassifier``."""


ROBERTA_CLASSIFIER_TRANSFORM = (
    lambda
        pretrained_roberta,
        max_sequence_length,
        truncation_strategy_title,
        truncation_strategy_text:
    Compose([
        BertTransform(
            tokenizer=RobertaTokenizer.from_pretrained(
                pretrained_roberta,
                do_lower_case=False),
            max_sequence_length=max_sequence_length,
            truncation_strategy=(
                truncation_strategy_title,
                truncation_strategy_text),
            starting_sep_token=True
        ),
        lambda d: {
            'input_ids': torch.tensor(d['input_ids']),
            'attention_mask': torch.tensor(d['input_mask'])
        }
    ])
)
"""The factory to create data transforms for ``RoBERTaClassifier``."""


# the RoBERTa ranking baseline

RoBERTaRanker = RobertaForMultipleChoice.from_pretrained
"""Rank choices with a softmax over a fine-tuned RoBERTa model."""


ROBERTA_RANKER_CONFIG = {
    'model': {
        # N.B. pretrained_model_name_or_path for the model must be the
        # same as pretrained_roberta for the transform
        'pretrained_model_name_or_path': 'roberta-large'
    },
    'transform': {
        # N.B. pretrained_roberta for the transform must be the same as
        # pretrained_model_name_or_path for the model
        'pretrained_roberta': 'roberta-large',
        'max_sequence_length': 90
    }
}
"""Configuration for ``RoBERTaRanker``."""


ROBERTA_RANKER_HYPER_PARAM_SPACE = [
    skopt.space.Real(
        low=1e-8,
        high=1e-2,
        prior='log-uniform',
        name='lr'),
    skopt.space.Real(
        low=1e-5,
        high=1e0,
        prior='log-uniform',
        name='weight_decay'),
    skopt.space.Real(
        low=0.0,
        high=1.0,
        prior='uniform',
        name='warmup_proportion'),
    skopt.space.Integer(
        low=1,
        high=25,
        name='n_epochs'),
    skopt.space.Integer(
        low=3,
        high=10,
        name='log_train_batch_size')
]
"""The hyper-param seach space for ``RoBERTaRanker``."""


ROBERTA_RANKER_TRANSFORM = (
    lambda
        pretrained_roberta,
        max_sequence_length:
    Compose([
        # wrap each action in a tuple for passing it to BertTransform
        lambda actions: tuple((action, None) for action in actions),
        # map BertTransform across all the action choices
        Map(
            transform=BertTransform(
                tokenizer=RobertaTokenizer.from_pretrained(
                    pretrained_roberta,
                    do_lower_case=False),
                max_sequence_length=max_sequence_length,
                truncation_strategy=('beginning', 'beginning'),
                starting_sep_token=True
            )
        ),
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
            ], dim=0)
        }
    ])
)
"""The factory to create data transforms for ``RoBERTaRanker``."""
