"""tests for scruples.utils."""

import logging
import math
import os
import tempfile
import unittest

import numpy as np
import pytest
from scipy import stats
from scipy.special import softmax

from scruples import utils


class ConfigureLoggingTestCase(unittest.TestCase):
    """Test scruples.utils.configure_logging."""

    def test_attaches_log_handler(self):
        n_handlers_before = len(logging.root.handlers)

        handler = utils.configure_logging()

        self.assertEqual(
            len(logging.root.handlers),
            n_handlers_before + 1)
        self.assertIn(handler, logging.root.handlers)

        logging.root.removeHandler(handler)

    def test_verbose_true_sets_log_level_to_debug(self):
        handler = utils.configure_logging(verbose=True)

        self.assertEqual(handler.level, logging.DEBUG)

        logging.root.removeHandler(handler)

    def test_verbose_false_sets_log_level_to_info(self):
        handler = utils.configure_logging(verbose=False)

        self.assertEqual(handler.level, logging.INFO)

        logging.root.removeHandler(handler)

    def test_verbose_defaults_to_false(self):
        handler = utils.configure_logging()

        self.assertEqual(handler.level, logging.INFO)

        logging.root.removeHandler(handler)


class CountWordsTestCase(unittest.TestCase):
    """Test scruples.utils.count_words."""

    def test_the_empty_string(self):
        self.assertEqual(utils.count_words(''), 0)

    def test_only_space(self):
        # test a single space
        self.assertEqual(utils.count_words(' '), 0)
        # test multiple kinds of spaces
        self.assertEqual(utils.count_words(' \n\t '), 0)

    def test_counting_one_word(self):
        # test a normal word
        self.assertEqual(utils.count_words('hello'), 1)
        # test a nonsense word
        self.assertEqual(utils.count_words('alkjef'), 1)

    def test_punctuation_is_stripped(self):
        # test punctuation with no words
        self.assertEqual(utils.count_words("?!"), 0)
        # test puctuation on the end of a word
        self.assertEqual(utils.count_words("Hey!"), 1)
        # test punctuation separated from a word
        self.assertEqual(utils.count_words("Hey !!!"), 1)

    def test_counting_words_separated_by_single_spaces(self):
        # test two words separated by a single space
        self.assertEqual(utils.count_words('two words'), 2)
        # test two words with punctuation
        self.assertEqual(utils.count_words('hello, world!'), 2)
        # test multiple words separated by single spaces
        self.assertEqual(utils.count_words('there are five words here.'), 5)

    def test_counting_words_separated_by_multiple_spaces(self):
        # test two words separated by multiple spaces
        self.assertEqual(utils.count_words('two   words'), 2)
        # test multiple words separated by multiple spaces
        self.assertEqual(
            utils.count_words('there   are\tfive    words.\nhere.'),
            5)

    def test_words_with_leading_and_trailing_space(self):
        # test leading space
        self.assertEqual(utils.count_words('   two words'), 2)
        # test trailing space
        self.assertEqual(utils.count_words('two words    '), 2)
        # test both leading and trailing space
        self.assertEqual(utils.count_words('    two words    '), 2)


class XentropyTestCase(unittest.TestCase):
    """Test scruples.utils.xentropy."""

    UNIFORM_1 = [1.]
    UNIFORM_2 = [0.5, 0.5]
    UNIFORM_5 = [0.2, 0.2, 0.2, 0.2, 0.2]
    SKEWED_2 = [0.7, 0.3]
    SKEWED_5 = [0.025, 0.075, 0.15, 0.25, 0.5]

    @staticmethod
    def _slow_xentropy(y_true, y_pred):
        return sum(
            - yt * math.log(yp)
            for yts, yps in zip(y_true, y_pred)
            for yt, yp in zip(yts, yps)
        ) / len(y_true)

    def test_xentropy_on_distribution_with_itself(self):
        # test using the uniform distribution on 1 symbol
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.UNIFORM_1],
                y_pred=[self.UNIFORM_1]),
            self._slow_xentropy(
                y_true=[self.UNIFORM_1],
                y_pred=[self.UNIFORM_1]))
        # test using the uniform distribution on 2 symbols
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.UNIFORM_2],
                y_pred=[self.UNIFORM_2]),
            self._slow_xentropy(
                y_true=[self.UNIFORM_2],
                y_pred=[self.UNIFORM_2]))
        # test using the uniform distribution on 5 symbols
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.UNIFORM_5],
                y_pred=[self.UNIFORM_5]),
            self._slow_xentropy(
                y_true=[self.UNIFORM_5],
                y_pred=[self.UNIFORM_5]))
        # test using a skewed distribution on 2 symbols
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.SKEWED_2],
                y_pred=[self.SKEWED_2]),
            self._slow_xentropy(
                y_true=[self.SKEWED_2],
                y_pred=[self.SKEWED_2]))
        # test using a skewed distribution on 5 symbols
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.SKEWED_5],
                y_pred=[self.SKEWED_5]),
            self._slow_xentropy(
                y_true=[self.SKEWED_5],
                y_pred=[self.SKEWED_5]))
        # test using a mix of distributions
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.UNIFORM_5, self.SKEWED_5],
                y_pred=[self.UNIFORM_5, self.SKEWED_5]),
            self._slow_xentropy(
                y_true=[self.UNIFORM_5, self.SKEWED_5],
                y_pred=[self.UNIFORM_5, self.SKEWED_5]))

    def test_xentropy_between_distributions(self):
        # test uniform as true and skewed as estimated
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.UNIFORM_5],
                y_pred=[self.SKEWED_5]),
            self._slow_xentropy(
                y_true=[self.UNIFORM_5],
                y_pred=[self.SKEWED_5]))
        # test using multiple samples
        self.assertAlmostEqual(
            utils.xentropy(
                y_true=[self.UNIFORM_5, self.UNIFORM_5],
                y_pred=[self.SKEWED_5, self.SKEWED_5]),
            self._slow_xentropy(
                y_true=[self.UNIFORM_5, self.UNIFORM_5],
                y_pred=[self.SKEWED_5, self.SKEWED_5]))

    def test_with_random_inputs(self):
        for _ in range(25):
            y_true = np.random.dirichlet(
                alpha=[1. for _ in range(7)],
                size=5)
            y_pred = np.random.dirichlet(
                alpha=[1. for _ in range(7)],
                size=5)
            self.assertAlmostEqual(
                utils.xentropy(y_true=y_true, y_pred=y_pred),
                self._slow_xentropy(y_true=y_true, y_pred=y_pred))


class MakeIdTestCase(unittest.TestCase):
    """Test scruples.utils.make_id."""

    def test_ids_are_uniq_with_high_probability(self):
        N_SAMPLES = 100000
        self.assertEqual(
            len(set(utils.make_id() for _ in range(N_SAMPLES))),
            N_SAMPLES)

    def test_ids_are_32_character_strings(self):
        for _ in range(5):
            id_ = utils.make_id()
            self.assertIsInstance(id_, str)
            self.assertEqual(len(id_), 32)


class MakeConfusionMatrixStrTestCase(unittest.TestCase):
    """Test scruples.utils.make_confusion_matrix_str."""

    def test_without_labels_argument(self):
        # test off diagonal entries are correct
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'b'],
                y_pred=['b', 'a']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      0 |      1 |\n'
            '| b      |      1 |      0 |\n'
            '+--------+--------+--------+\n'
            ' true')
        # test on diagonal entries are correct
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'b'],
                y_pred=['a', 'b']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      1 |      0 |\n'
            '| b      |      0 |      1 |\n'
            '+--------+--------+--------+\n'
            ' true')
        # test on and off diagonal entries simultaneously
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'b', 'a', 'b'],
                y_pred=['a', 'b', 'b', 'a']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      1 |      1 |\n'
            '| b      |      1 |      1 |\n'
            '+--------+--------+--------+\n'
            ' true')
        # test that y_true corresponds to the 'true' label and y_pred
        # corresponds to the 'predicted' label in the output
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'a', 'a', 'a'],
                y_pred=['a', 'b', 'b', 'a']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      2 |      2 |\n'
            '| b      |      0 |      0 |\n'
            '+--------+--------+--------+\n'
            ' true')

    def test_with_labels_argument(self):
        # test off diagonal entries are correct
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'b', 'c'],
                y_pred=['b', 'a', 'b'],
                labels=['a', 'b']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      0 |      1 |\n'
            '| b      |      1 |      0 |\n'
            '+--------+--------+--------+\n'
            ' true')
        # test on diagonal entries are correct
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'b', 'c'],
                y_pred=['a', 'b', 'c'],
                labels=['a', 'b']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      1 |      0 |\n'
            '| b      |      0 |      1 |\n'
            '+--------+--------+--------+\n'
            ' true')
        # test on and off diagonal entries simultaneously
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'b', 'a', 'b', 'c', 'd'],
                y_pred=['a', 'b', 'b', 'a', 'd', 'c'],
                labels=['a', 'b']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      1 |      1 |\n'
            '| b      |      1 |      1 |\n'
            '+--------+--------+--------+\n'
            ' true')
        # test that y_true corresponds to the 'true' label and y_pred
        # corresponds to the 'predicted' label in the output
        self.assertEqual(
            utils.make_confusion_matrix_str(
                y_true=['a', 'a', 'a', 'a', 'c', 'd'],
                y_pred=['a', 'b', 'b', 'a', 'd', 'c'],
                labels=['a', 'b']),
            '+========+========+========+\n'
            '|        | a      | b      | predicted\n'
            '+========+========+========+\n'
            '| a      |      2 |      2 |\n'
            '| b      |      0 |      0 |\n'
            '+--------+--------+--------+\n'
            ' true')


class MakeLabelDistributionStrTestCase(unittest.TestCase):
    """Test scruples.utils.make_label_distribution_str."""

    def test_without_labels_argument(self):
        # test only one label
        self.assertEqual(
            utils.make_label_distribution_str(y_true=['a', 'a', 'a', 'a']),
            '+==========+==========+\n'
            '|          | a        |\n'
            '+==========+==========+\n'
            '| fraction |   1.0000 |\n'
            '+----------+----------+\n'
            '| total    |        4 |\n'
            '+----------+----------+\n')
        # test multiple labels
        self.assertEqual(
            utils.make_label_distribution_str(y_true=['a', 'a', 'b', 'c']),
            '+==========+==========+==========+==========+\n'
            '|          | a        | b        | c        |\n'
            '+==========+==========+==========+==========+\n'
            '| fraction |   0.5000 |   0.2500 |   0.2500 |\n'
            '+----------+----------+----------+----------+\n'
            '| total    |        2 |        1 |        1 |\n'
            '+----------+----------+----------+----------+\n')

    def test_with_labels_argument(self):
        # test only one label
        self.assertEqual(
            utils.make_label_distribution_str(
                y_true=['a', 'a', 'a', 'a'],
                labels=['a', 'b']),
            '+==========+==========+==========+\n'
            '|          | a        | b        |\n'
            '+==========+==========+==========+\n'
            '| fraction |   1.0000 |   0.0000 |\n'
            '+----------+----------+----------+\n'
            '| total    |        4 |        0 |\n'
            '+----------+----------+----------+\n')
        # test multiple labels
        self.assertEqual(
            utils.make_label_distribution_str(
                y_true=['a', 'a', 'b', 'c'],
                labels=['a', 'b']),
            '+==========+==========+==========+\n'
            '|          | a        | b        |\n'
            '+==========+==========+==========+\n'
            '| fraction |   0.5000 |   0.2500 |\n'
            '+----------+----------+----------+\n'
            '| total    |        2 |        1 |\n'
            '+----------+----------+----------+\n')


class NextUniquePathTestCase(unittest.TestCase):
    """Test scruples.utils.next_unique_path."""

    def test_when_path_is_already_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'foo')

            self.assertEqual(utils.next_unique_path(path), path)

    def test_when_path_is_not_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'foo')
            path1 = os.path.join(temp_dir, 'foo_1')
            path2 = os.path.join(temp_dir, 'foo_2')

            # when path is not unique
            with open(path, 'w') as f:
                f.write('')

            self.assertEqual(utils.next_unique_path(path), path1)

            # when the first modification of path is still not unique
            with open(path1, 'w') as f:
                f.write('')

            self.assertEqual(utils.next_unique_path(path), path2)


class EstimateBetaBinomialParametersTestCase(unittest.TestCase):
    """Test scruples.utils.estimate_beta_binomial_parameters."""

    @pytest.mark.slow
    def test_estimates_parameters_for_beta_binomial_with_fixed_n(self):
        a_bs = [
            (1, 1), # uniform prior (beta(1, 1))
            (5, 0.5), # right-skewed prior (beta(5, 0.5))
            (0.5, 5) # left-skewed prior (beta(0.5, 5))
        ]
        for a, b in a_bs:
            n = 100
            ps = stats.beta.rvs(a=a, b=b, size=100000)
            ss = stats.binom.rvs(n, ps)
            fs = n - ss

            a_hat, b_hat = utils.estimate_beta_binomial_parameters(
                successes=ss, failures=fs)

            self.assertLess(abs(a_hat - a), 0.1)
            self.assertLess(abs(b_hat - b), 0.1)

    @pytest.mark.slow
    def test_estimates_parameters_for_beta_binomial_with_random_n(self):
        a_bs = [
            (1, 1), # uniform prior (beta(1, 1))
            (5, 0.5), # right-skewed prior (beta(5, 0.5))
            (0.5, 5) # left-skewed prior (beta(0.5, 5))
        ]
        for a, b in a_bs:
            ps = stats.beta.rvs(a=a, b=b, size=100000)
            ns = np.random.choice(range(50, 151), size=100000)
            ss = stats.binom.rvs(ns, ps)
            fs = ns - ss

            a_hat, b_hat = utils.estimate_beta_binomial_parameters(
                successes=ss, failures=fs)

            self.assertLess(abs(a_hat - a), 0.1)
            self.assertLess(abs(b_hat - b), 0.1)


class EstimateDirichletMultinomialParametersTestCase(unittest.TestCase):
    """Test scruples.utils.estimate_dirichlet_multinomial_parameters."""

    @pytest.mark.slow
    def test_estimates_parameters_for_dirichlet_multinomial_with_fixed_n(self):
        paramss = [
            # uniform priors
            (1, 1),
            (1, 1, 1),
            (1, 1, 1, 1),
            # skewed priors
            (5, 0.5),
            (0.5, 5),
            (0.5, 5, 5),
            (5, 0.5, 5),
            (5, 5, 0.5)
        ]
        for params in paramss:
            n = 100
            pss = stats.dirichlet.rvs(alpha=params, size=100000)
            obs = np.array([
                stats.multinomial.rvs(n, ps)
                for ps in pss
            ])

            params_hat = utils.estimate_dirichlet_multinomial_parameters(
                observations=obs)

            self.assertTrue(np.allclose(params_hat, params, atol=0.1))

    @pytest.mark.slow
    def test_estimates_parameters_for_dirichlet_multinomial_with_random_n(self):
        paramss = [
            # uniform priors
            (1, 1),
            (1, 1, 1),
            (1, 1, 1, 1),
            # skewed priors
            (5, 0.5),
            (0.5, 5),
            (0.5, 5, 5),
            (5, 0.5, 5),
            (5, 5, 0.5)
        ]
        for params in paramss:
            ns = np.random.choice(range(50, 151), size=100000)
            pss = stats.dirichlet.rvs(alpha=params, size=100000)
            obs = np.array([
                stats.multinomial.rvs(n, ps)
                for n, ps in zip(ns, pss)
            ])
            params_hat = utils.estimate_dirichlet_multinomial_parameters(
                observations=obs)

            self.assertTrue(np.allclose(params_hat, params, atol=0.1))


class CalibrationFactorTestCase(unittest.TestCase):
    """Test scruples.utils.calibration_factor."""

    def test_calibration_factor(self):
        for _ in range(100):
            logits = np.random.randn(100, 3)
            targets = softmax(
                logits + np.random.rand(1) * np.random.randn(100, 3),
                axis=-1)

            t_opt = utils.calibration_factor(logits=logits, targets=targets)

            ts = np.linspace(max(1e-10, t_opt - 10), t_opt + 10, num=21)
            for t in ts:
                self.assertLess(
                    utils.xentropy(y_true=targets, y_pred=softmax(logits / t_opt, axis=-1)),
                    utils.xentropy(y_true=targets, y_pred=softmax(logits / t, axis=-1)))
