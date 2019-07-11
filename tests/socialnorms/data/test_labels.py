"""Tests for scruples.data.labels."""

import unittest

from scruples.data import labels


class LabelTestCase(unittest.TestCase):
    """Test scruples.data.labels.Label."""

    LABEL_TO_PHRASES = {
        labels.Label.AUTHOR: [
            "you're the asshole",
            'your the asshole',
            'you are the asshole',
            "you're the ahole",
            'your the ahole',
            'you are the ahole',
            "you're the a-hole",
            "you're kind of an asshole",
            "you're a huge asshole",
            "you are indeed an asshole",
            "you're just a big asshole"
        ],
        labels.Label.OTHER: [
            'not an asshole',
            'not the asshole',
            'not an ahole',
            'not the ahole',
            'not an a-hole',
            'not really an asshole'
        ],
        labels.Label.EVERYBODY: [
            'everyone sucks here',
            'everybody sucks here',
            'every one sucks here',
            'every body sucks here',
            'you both suck'
        ],
        labels.Label.NOBODY: [
            'no assholes here',
            'no aholes here',
            'no a-holes here'
        ],
        labels.Label.INFO: [
            'not enough info',
            'need more info',
            'needs more info',
            "more info's required",
            'more info is required',
            'more info required'
        ]
    }
    """A dictionary mapping labels to phrases that signify them."""

    # test extract_from_text

    def test_extract_from_text_on_initialisms(self):
        for label in labels.Label:
            # when the initialism starts the text
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'{label.reddit_name}. Definitely.'),
                label)
            # when the initialism ends the text
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.reddit_name}'),
                label)
            # when the initialism is in the middle of the text
            self.assertEqual(
                labels.Label.extract_from_text(
                    f"I'd say {label.reddit_name}, but I'm not sure."),
                label)
            # when the initialism is uppercased
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.reddit_name.upper()}.'),
                label)
            # when the initialism is lowercased
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.reddit_name.lower()}.'),
                label)
            # when the initialism is capitalized
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.reddit_name.lower().capitalize()}.'),
                label)

    def test_extract_from_text_on_phrases(self):
        for label, phrases in self.LABEL_TO_PHRASES.items():
            for phrase in phrases:
                # when the phrase starts the text
                self.assertEqual(
                    labels.Label.extract_from_text(
                        f"{phrase}, I'm pretty sure."),
                    label)
                # when the phrase ends the text
                self.assertEqual(
                    labels.Label.extract_from_text(
                        f"I'm pretty sure {phrase}"),
                    label)
                # when the phrase is in the middle of the text
                self.assertEqual(
                    labels.Label.extract_from_text(
                        f"I think {phrase}, but I'm not sure."),
                    label)
                # when the phrase is uppercased
                self.assertEqual(
                    labels.Label.extract_from_text(
                        f'I think {phrase.upper()}.'),
                    label)
                # when the phrase is lowercased
                self.assertEqual(
                    labels.Label.extract_from_text(
                        f'I think {phrase.lower()}.'),
                    label)
                # when the phrase is capitalized
                self.assertEqual(
                    labels.Label.extract_from_text(
                        f'{phrase.lower().capitalize()}, I think.'),
                    label)

    def test_extract_from_text_on_text_denoting_multiple_labels(self):
        self.assertEqual(
            labels.Label.extract_from_text(
                "YTA. They're totally NTA here!").name,
            'AUTHOR')
        self.assertEqual(
            labels.Label.extract_from_text(
                "I was going to say YTA, but thinking more I'll say NTA."
            ).name,
            'OTHER')

    def test_extract_from_text_doesnt_extract_spurious_labels(self):
        self.assertEqual(
            labels.Label.extract_from_text("Interesting story."),
            None)

    # test find

    def test_find_on_initialisms(self):
        for label1 in labels.Label:
            for label2 in labels.Label:
                if label1 == label2:
                    continue
                # when the initialism starts the text
                self.assertEqual(
                    label1.find(f'{label1.reddit_name}. Definitely.'),
                    (0, len(label1.reddit_name)))
                self.assertIsNone(
                    label2.find(f'{label1.reddit_name}. Definitely.'))
                # when the initialism ends the text
                self.assertEqual(
                    label1.find(f'I think {label1.reddit_name}'),
                    (8, 8 + len(label1.reddit_name)))
                self.assertIsNone(
                    label2.find(f'I think {label1.reddit_name}'))
                # when the initialism is in the middle of the text
                self.assertEqual(
                    label1.find(
                        f"I'd say {label1.reddit_name}, but I'm not sure."),
                    (8, 8 + len(label1.reddit_name)))
                self.assertIsNone(
                    label2.find(
                        f"I'd say {label1.reddit_name}, but I'm not sure."))
                # when the initialism is uppercased
                self.assertEqual(
                    label1.find(f'I think {label1.reddit_name.upper()}.'),
                    (8, 8 + len(label1.reddit_name)))
                self.assertIsNone(
                    label2.find(f'I think {label1.reddit_name.upper()}.'))
                # when the initialism is lowercased
                self.assertEqual(
                    label1.find(f'I think {label1.reddit_name.lower()}.'),
                    (8, 8 + len(label1.reddit_name)))
                self.assertIsNone(
                    label2.find(f'I think {label1.reddit_name.lower()}.'))
                # when the initialism is capitalized
                self.assertEqual(
                    label1.find(
                        f'I think {label1.reddit_name.lower().capitalize()}.'),
                    (8, 8 + len(label1.reddit_name)))
                self.assertIsNone(
                    label2.find(
                        f'I think {label1.reddit_name.lower().capitalize()}.'))

    def test_find_on_phrases(self):
        for label1 in labels.Label:
            for label2 in labels.Label:
                if label1 == label2:
                    continue
                for phrase in self.LABEL_TO_PHRASES[label1]:
                    # when the phrase starts the text
                    self.assertEqual(
                        label1.find(f"{phrase}, I'm pretty sure."),
                        (0, len(phrase)))
                    self.assertIsNone(
                        label2.find(f"{phrase}, I'm pretty sure."))
                    # when the phrase ends the text
                    self.assertEqual(
                        label1.find(f"I'm pretty sure {phrase}"),
                        (16, 16 + len(phrase)))
                    self.assertIsNone(
                        label2.find(f"I'm pretty sure {phrase}"))
                    # when the phrase is in the middle of the text
                    self.assertEqual(
                        label1.find(f"I think {phrase}, but I'm not sure."),
                        (8, 8 + len(phrase)))
                    self.assertIsNone(
                        label2.find(f"I think {phrase}, but I'm not sure."))
                    # when the phrase is uppercased
                    self.assertEqual(
                        label1.find(f'I think {phrase.upper()}.'),
                        (8, 8 + len(phrase)))
                    self.assertIsNone(
                        label2.find(f'I think {phrase.upper()}.'))
                    # when the phrase is lowercased
                    self.assertEqual(
                        label1.find(f'I think {phrase.lower()}.'),
                        (8, 8 + len(phrase)))
                    self.assertIsNone(
                        label2.find(f'I think {phrase.lower()}.'))
                    # when the phrase is capitalized
                    self.assertEqual(
                        label1.find(f'I think {phrase.lower().capitalize()}.'),
                        (8, 8 + len(phrase)))
                    self.assertIsNone(
                        label2.find(f'I think {phrase.lower().capitalize()}.'))

    def test_find_doesnt_return_true_on_spurious_text(self):
        for label in labels.Label:
            self.assertIsNone(label.find('Empty text.'))

    def test_find_returns_the_first_span_denoting_the_label(self):
        for label in labels.Label:
            # when the utterance is an initialism
            self.assertEqual(
                label.find(
                    f'{label.reddit_name} foo {label.reddit_name}.'),
                (0, len(label.reddit_name)))
            self.assertEqual(
                label.find(
                    f'foo {label.reddit_name} bar {label.reddit_name}.'),
                (4, 4 + len(label.reddit_name)))
            self.assertEqual(
                label.find(
                    f'foo {label.reddit_name} bar {label.reddit_name}'
                    f' baz {label.reddit_name}.'),
                (4, 4 + len(label.reddit_name)))
            # when the utterance is a phrase
            for phrase in self.LABEL_TO_PHRASES[label]:
                self.assertEqual(
                    label.find(f'{phrase} foo {phrase}.'),
                    (0, len(phrase)))
                self.assertEqual(
                    label.find(f'foo {phrase} bar {phrase}.'),
                    (4, 4 + len(phrase)))
                self.assertEqual(
                    label.find(f'foo {phrase} bar {phrase} baz {phrase}.'),
                    (4, 4 + len(phrase)))
