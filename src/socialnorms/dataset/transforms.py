"""Transformations for reading the data."""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple)

from pytorch_pretrained_bert.tokenization import BertTokenizer


class BertTransform(object):
    """Transform a tuple of text into input for BERT.

    Parameters
    ----------
    tokenizer : BertTokenizer, required
        The ``BertTokenizer`` instance to use for tokenization.
    max_sequence_length : int, optional (default=512)
        The maximum length the output sequence should be. If the input
        converts to something longer, it will be truncated. The max
        sequence length must be shorter than the maximum sequence length
        the BERT model accepts (512 for the standard pretrained BERT).
    truncation_strategy : Tuple[(str, str)],
                          optional (default=("beginning", "beginning"))
        A tuple of strings providing the truncation strategy to use on
        each of the two pieces of text. Each string must be one of
        ``"beginning"``, or ``"ending"``. These strings correspond to
        taking text from the beginning of the input, or the ending of
        the input, respectively.
    """
    TRUNCATION_STRATEGIES = ['beginning', 'ending']

    def __init__(
            self,
            tokenizer: BertTokenizer,
            max_sequence_length: int = 512,
            truncation_strategy: Tuple[str, str] = ('beginning', 'beginning')
    ) -> None:
        if not isinstance(tokenizer, BertTokenizer):
            raise ValueError('tokenizer must be an instance of BertTokenizer')

        max_sequence_length = int(max_sequence_length)
        if max_sequence_length < 5:
            # if each piece of text only has 1 token, then the sequence
            # length will be 5 including separator tokens.
            raise ValueError(
                'max_sequence_length must be greater than or equal to'
                ' 5')

        truncation_strategy = tuple(truncation_strategy)
        for strategy in truncation_strategy:
            if strategy not in self.TRUNCATION_STRATEGIES:
                raise ValueError(
                    f'truncation strategy {strategy} does not'
                    f' exist. Please use one of'
                    f' {", ".join(self.TRUNCATION_STRATEGIES)}.')

        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.truncation_strategy = truncation_strategy

    def __call__(
            self,
            feature: Tuple[str, Optional[str]]
    ) -> Dict[str, List[int]]:
        """Return ``feature`` transformed into BERT inputs.

        Parameters
        ----------
        feature : Tuple[str, Optional[str]]
            The tuple of text to transform into BERT inputs. If the
            second string is ``None``, then only the first string will
            be encoded.

        Returns
        -------
        input_ids : List[int]
            The list of IDs for the tokenized word pieces.
        input_mask : List[int]
            A 0-1 mask for ``input_ids``.
        segment_ids : List[int]
            IDs for which segment each wordpiece comes from (the first
            or second string in the pair).

        Output is returned as a dictionary.
        """
        # tokenize the text
        feature = (
            self.tokenizer.tokenize(feature[0]),
            self.tokenizer.tokenize(feature[1])
                if feature[1] is not None
                else None
        )

        # truncate the token sequences
        feature = self._truncate(feature)

        # convert the tokens to input ids, input mask, and segment ids.
        tokens = ['[CLS]'] + feature[0] + ['[SEP]']
        segment_ids = [0 for _ in range(len(tokens))]
        if feature[1]:
            tokens += feature[1] + ['[SEP]']
            segment_ids += [1 for _ in range(len(feature[1]) + 1)]

        padding = [0 for _ in range(self.max_sequence_length - len(tokens))]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) + padding
        input_mask = [1 for _ in range(len(tokens))] + padding
        segment_ids = segment_ids + padding

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }

    def _truncate(
            self,
            feature: Tuple[List[str], Optional[List[str]]]
    ) -> Tuple[List[str], Optional[List[str]]]:
        """Return ``feature`` truncated to ``self.max_sequence_length``.

        Parameters
        ----------
        feature : Tuple[List[str], Optional[List[str]]]
            The tuple of tokenized strings for the instance.

        Returns
        -------
        Tuple[List[str], Optional[List[str]]]
            The truncated tokenized strings for the instance.
        """
        feature0, feature1 = feature
        strategy0, strategy1 = self.truncation_strategy

        # compute the number of tokens to keep from each sequence
        if feature1 is None:
            tokens_to_keep0 = self.max_sequence_length - 2
            tokens_to_keep1 = None
        else:
            tokens_to_keep0 = max(
                (self.max_sequence_length - 3) // 2,
                self.max_sequence_length - len(feature1) - 3)
            tokens_to_keep1 = max(
                (self.max_sequence_length - 3) // 2,
                self.max_sequence_length - len(feature0) - 3)

        if strategy0 == 'beginning':
            feature0 = feature0[:tokens_to_keep0]
        elif strategy0 == 'ending':
            feature0 = feature0[-tokens_to_keep0:]
        else:
            raise ValueError(
                f'Unrecognized truncation_strategy {strategy0}.')

        if feature1 is None:
            pass
        elif strategy1 == 'beginning':
            feature1 = feature1[:tokens_to_keep1]
        elif strategy1 == 'ending':
            feature1 = feature1[-tokens_to_keep1:]
        else:
            raise ValueError(
                f'Unrecognized truncation_strategy {strategy1}.')

        return feature0, feature1


class Compose(object):
    """Compose a sequence of transforms.

    Parameters
    ----------
    transforms : Sequence[Callable]
        The sequence of transforms to compose. The transforms should be
        in pipeline order, i.e. the first transform to apply goes first
        in the list.
    """
    def __init__(
            self,
            transforms: Sequence[Callable]
    ) -> None:
        self.transforms = transforms

    def __call__(
            self,
            feature: Any
    ) -> Any:
        """Return ``feature`` with all the transforms applied.

        Returns
        -------
        Any
            The result of applying all the transforms to ``feature``.
        """
        for transform in self.transforms:
            feature = transform(feature)

        return feature


class Map(object):
    """Map a transform across a sequence.

    Parameters
    ----------
    transform : Callable
        The transform to map across the sequence.
    """
    def __init__(
            self,
            transform: Callable
    ) -> None:
        self.transform = transform

    def __call__(
            self,
            feature: Sequence[Any]
    ) -> List[Any]:
        """Return ``feature`` mapped by ``self.transform``.

        Return ``feature`` mapped by ``self.transform`` and cast to a
        list.

        Returns
        -------
        List[Any]
            ``feature`` mapped by the transform with which this object
            was initialized, and then cast to a list.
        """
        return [self.transform(x) for x in feature]
