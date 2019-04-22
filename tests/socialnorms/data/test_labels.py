"""Tests for socialnorms.data.labels."""

import unittest

from socialnorms.data import labels


class LabelTestCase(unittest.TestCase):
    """Test socialnorms.data.labels.Label."""

    LABEL_TO_PHRASES = {
        labels.Label.YTA: [
            "you're the asshole",
            'your the asshole',
            'you are the asshole',
            "you're the ahole",
            'your the ahole',
            'you are the ahole',
            "you're the a-hole",
            "you're kind of an asshole"
        ],
        labels.Label.NTA: [
            'not an asshole',
            'not the asshole',
            'not an ahole',
            'not the ahole',
            'not an a-hole',
            'not really an asshole'
        ],
        labels.Label.ESH: [
            'everyone sucks here',
            'everybody sucks here',
            'every one sucks here',
            'every body sucks here'
        ],
        labels.Label.NAH: [
            'no assholes here',
            'no aholes here',
            'no a-holes here'
        ],
        labels.Label.INFO: [
            'not enough info',
            'need more info',
            'needs more info'
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
                labels.Label.extract_from_text(f'{label.name}. Definitely.'),
                label)
            # when the initialism ends the text
            self.assertEqual(
                labels.Label.extract_from_text(f'I think {label.name}'),
                label)
            # when the initialism is in the middle of the text
            self.assertEqual(
                labels.Label.extract_from_text(
                    f"I'd say {label.name}, but I'm not sure."),
                label)
            # when the initialism is uppercased
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.name.upper()}.'),
                label)
            # when the initialism is lowercased
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.name.lower()}.'),
                label)
            # when the initialism is capitalized
            self.assertEqual(
                labels.Label.extract_from_text(
                    f'I think {label.name.lower().capitalize()}.'),
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

    def test_extract_from_text_on_ambiguous_cases(self):
        self.assertEqual(
            labels.Label.extract_from_text(
                "YTA, but maybe NTA... I'm not sure."),
            None)

    def test_extract_from_text_doesnt_extract_spurious_labels(self):
        self.assertEqual(
            labels.Label.extract_from_text("Interesting story."),
            None)

    # test in_

    def test_in__on_initialisms(self):
        for label1 in labels.Label:
            for label2 in labels.Label:
                if label1 == label2:
                    continue
                # when the initialism starts the text
                self.assertTrue(
                    label1.in_(f'{label1.name}. Definitely.'))
                self.assertFalse(
                    label2.in_(f'{label1.name}. Definitely.'))
                # when the initialism ends the text
                self.assertTrue(
                    label1.in_(f'I think {label1.name}'))
                self.assertFalse(
                    label2.in_(f'I think {label1.name}'))
                # when the initialism is in the middle of the text
                self.assertTrue(
                    label1.in_(f"I'd say {label1.name}, but I'm not sure."))
                self.assertFalse(
                    label2.in_(f"I'd say {label1.name}, but I'm not sure."))
                # when the initialism is uppercased
                self.assertTrue(
                    label1.in_(f'I think {label1.name.upper()}.'))
                self.assertFalse(
                    label2.in_(f'I think {label1.name.upper()}.'))
                # when the initialism is lowercased
                self.assertTrue(
                    label1.in_(f'I think {label1.name.lower()}.'))
                self.assertFalse(
                    label2.in_(f'I think {label1.name.lower()}.'))
                # when the initialism is capitalized
                self.assertTrue(
                    label1.in_(f'I think {label1.name.lower().capitalize()}.'))
                self.assertFalse(
                    label2.in_(f'I think {label1.name.lower().capitalize()}.'))

    def test_in__on_phrases(self):
        for label1 in labels.Label:
            for label2 in labels.Label:
                if label1 == label2:
                    continue
                for phrase in self.LABEL_TO_PHRASES[label1]:
                    # when the phrase starts the text
                    self.assertTrue(
                        label1.in_(f"{phrase}, I'm pretty sure."))
                    self.assertFalse(
                        label2.in_(f"{phrase}, I'm pretty sure."))
                    # when the phrase ends the text
                    self.assertTrue(
                        label1.in_(f"I'm pretty sure {phrase}"))
                    self.assertFalse(
                        label2.in_(f"I'm pretty sure {phrase}"))
                    # when the phrase is in the middle of the text
                    self.assertTrue(
                        label1.in_(f"I think {phrase}, but I'm not sure."))
                    self.assertFalse(
                        label2.in_(f"I think {phrase}, but I'm not sure."))
                    # when the phrase is uppercased
                    self.assertTrue(
                        label1.in_(f'I think {phrase.upper()}.'))
                    self.assertFalse(
                        label2.in_(f'I think {phrase.upper()}.'))
                    # when the phrase is lowercased
                    self.assertTrue(
                        label1.in_(f'I think {phrase.lower()}.'))
                    self.assertFalse(
                        label2.in_(f'I think {phrase.lower()}.'))
                    # when the phrase is capitalized
                    self.assertTrue(
                        label1.in_(f'I think {phrase.lower().capitalize()}.'))
                    self.assertFalse(
                        label2.in_(f'I think {phrase.lower().capitalize()}.'))

    def test_in__doesnt_return_true_on_spurious_text(self):
        for label in labels.Label:
            self.assertFalse(label.in_('Empty text.'))
