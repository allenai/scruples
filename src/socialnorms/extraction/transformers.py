"""Transformations for extracting data."""

from pattern.en import (
    conjugate,
    PRESENT,
    INDICATIVE,
    PROGRESSIVE)
import spacy

from . import base


# TODO: remove the following lines once this issue:
# https://github.com/clips/pattern/issues/243, is resolved.
try: conjugate('give')
except: pass


class GerundifyingTransformer(base.LoggedCallable):
    """Transform the input into a gerund phrase."""

    _nlp = spacy.load('en', disable=['ner'])

    @classmethod
    def _is_root(
            cls,
            token: spacy.tokens.token.Token
    ) -> bool:
        return token.dep_ == 'ROOT' and token.pos_ == 'VERB'

    @classmethod
    def _is_rootlike(
            cls,
            token: spacy.tokens.token.Token
    ) -> bool:
        return (
            cls._is_root(token)
            or any(cls._is_root(c) for c in token.conjuncts)
        )

    @classmethod
    def _conjugate(
            cls,
            text: str
    ) -> str:
        # handle some special cases
        if text in ["'m", 'm']:
            text = 'am'
        elif text == 'left':
            text = 'leave'
        else:
            pass

        return conjugate(
            verb=text,
            tense=PRESENT,
            person=None,
            number=None,
            mood=INDICATIVE,
            aspect=PROGRESSIVE)

    def apply(
            self,
            x: str
    ) -> str:
        token_strs = []
        for token in self._nlp(x):
            if token.sent.start != 0:
                # skip conjugating verbs that aren't in the first
                # sentence
                token_strs.append(token.text_with_ws)
                continue

            if self._is_rootlike(token):
                # conjugate the token if it is like the root verb
                token_str = self._conjugate(token.text)\
                    + token.text_with_ws[len(token.text):]
            elif token.dep_ == 'nsubj' and self._is_rootlike(token.head):
                # remove the subject attached to the root
                continue
            elif token.dep_ == 'aux' and self._is_rootlike(token.head):
                # remove auxiliary verbs attached to the root
                continue
            elif token.text in ["n't", 'nt'] and self._is_rootlike(token.head):
                # fix forms of "not" coming from contractions
                token_str = 'not' + token.text_with_ws[len(token.text):]
            else:
                # nothing to do here
                token_str = token.text_with_ws

            if (
                    token_str.startswith('not')
                    and len(token_strs) > 0
                    and token_strs[-1] in ['being', 'doing', 'having']
            ):
                # swap "not" with the preceding verb
                token_str, token_strs[-1] = token_strs[-1], token_str

            token_strs.append(token_str)

        return ''.join(token_strs)
