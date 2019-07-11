"""Tests for scruples.dataset.readers."""

import os
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

import scruples.settings as scruples_settings
from scruples.dataset import readers
from ... import settings, utils


class ScruplesCorpusTestCase(unittest.TestCase):
    """Test scruples.dataset.readers.ScruplesCorpus."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # copy the corpus-easy fixture to the temporary directory
        for split in readers.ScruplesCorpus.SPLITS:
            split_filename =\
                scruples_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(settings.CORPUS_EASY_DIR, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_corpus_has_correct_splits(self):
        self.assertEqual(
            set(readers.ScruplesCorpus.SPLITS),
            set(['train', 'dev', 'test']))

    def test_reads_in_splits(self):
        # read the dataset
        corpus = readers.ScruplesCorpus(data_dir=self.temp_dir.name)

        # train
        train_ids, train_features, train_labels = corpus.train
        self.assertIsInstance(train_ids, pd.Series)
        self.assertEqual(train_ids.tolist()[0], 'id_0')
        self.assertIsInstance(train_features, pd.DataFrame)
        self.assertEqual(
            train_features.to_dict(orient='records')[0],
            {
                "title": "AITA test post",
                "text": "This post is nta."
            })
        self.assertIsInstance(train_labels, pd.Series)
        self.assertEqual(train_labels.tolist()[0], 'OTHER')

        # dev
        dev_ids, dev_features, dev_labels = corpus.dev
        self.assertIsInstance(dev_ids, pd.Series)
        self.assertEqual(dev_ids.tolist()[0], 'id_20')
        self.assertIsInstance(dev_features, pd.DataFrame)
        self.assertEqual(
            dev_features.to_dict(orient='records')[0],
            {
                "title": "AITA test post",
                "text": "Label this post nta."
            })
        self.assertIsInstance(dev_labels, pd.Series)
        self.assertEqual(dev_labels.tolist()[0], 'OTHER')

        # test
        test_ids, test_features, test_labels = corpus.test
        self.assertIsInstance(test_ids, pd.Series)
        self.assertEqual(test_ids.tolist()[0], 'id_25')
        self.assertIsInstance(test_features, pd.DataFrame)
        self.assertEqual(
            test_features.to_dict(orient='records')[0],
            {
                "title": "AITA test post",
                "text": "The label for this post should be nta."
            })
        self.assertIsInstance(test_labels, pd.Series)
        self.assertEqual(test_labels.tolist()[0], 'OTHER')


class ScruplesCorpusDatasetTestCase(unittest.TestCase):
    """Test scruples.dataset.readers.ScruplesCorpusDataset."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # copy the corpus-easy dataset to the temporary directory
        for split in readers.ScruplesCorpusDataset.SPLITS:
            split_filename =\
                scruples_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(settings.CORPUS_EASY_DIR, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_corpus_has_correct_splits(self):
        self.assertEqual(
            set(readers.ScruplesCorpusDataset.SPLITS),
            set(['train', 'dev', 'test']))

    def test_init_raises_error_if_bad_split_provided(self):
        with self.assertRaisesRegex(
                ValueError,
                r'split must be one of train, dev, test\.'
        ):
          readers.ScruplesCorpusDataset(
              data_dir=self.temp_dir.name,
              split='val',
              transform=None,
              label_transform=None)

    def test_len(self):
        # test len on train
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None)
        self.assertEqual(len(train), 20)

        # test len on dev
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None)
        self.assertEqual(len(dev), 5)

        # test len on test
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None)
        self.assertEqual(len(test), 5)

    def test___get_item__(self):
        # test __get_item__ on train
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None)
        id_, feature, label = train[0]
        self.assertEqual(id_, 'id_0')
        self.assertEqual(feature, (
            "AITA test post",
            "This post is nta."
        ))
        self.assertEqual(label, 'OTHER')

        # test __get_item__ on dev
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None)
        id_, feature, label = dev[0]
        self.assertEqual(id_, 'id_20')
        self.assertEqual(feature, (
            "AITA test post",
            "Label this post nta."
        ))
        self.assertEqual(label, 'OTHER')

        # test __get_item__ on test
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None)
        id_, feature, label = test[0]
        self.assertEqual(id_, 'id_25')
        self.assertEqual(feature, (
            "AITA test post",
            "The label for this post should be nta."
        ))
        self.assertEqual(label, 'OTHER')

    def test_init_with_transform(self):
        # test the train split
        train_transform = Mock(return_value='foo')
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=train_transform,
            label_transform=None)
        # call the transform and check the return value
        _, feature, _ = train[0]
        self.assertEqual(feature, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_transform.call_count, 1)
        args, kwargs = train_transform.call_args
        self.assertEqual(args, ((
            "AITA test post",
            "This post is nta."
        ),))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_transform = Mock(return_value='foo')
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=dev_transform,
            label_transform=None)
        # call the transform and check the return value
        _, feature, _ = dev[0]
        self.assertEqual(feature, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_transform.call_count, 1)
        args, kwargs = dev_transform.call_args
        self.assertEqual(args, ((
            "AITA test post",
            "Label this post nta."
        ),))
        self.assertEqual(kwargs, {})

        # test the test split
        test_transform = Mock(return_value='foo')
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=test_transform,
            label_transform=None)
        # call the transform and check the return value
        _, feature, _ = test[0]
        self.assertEqual(feature, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_transform.call_count, 1)
        args, kwargs = test_transform.call_args
        self.assertEqual(args, ((
            "AITA test post",
            "The label for this post should be nta."
        ),))
        self.assertEqual(kwargs, {})

    def test_init_with_label_transform(self):
        # test the train split
        train_label_transform = Mock(return_value='foo')
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=train_label_transform)
        # call the transform and check the return value
        _, _, label = train[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_label_transform.call_count, 1)
        args, kwargs = train_label_transform.call_args
        self.assertEqual(args, ('OTHER',))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_label_transform = Mock(return_value='foo')
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=dev_label_transform)
        # call the transform and check the return value
        _, _, label = dev[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_label_transform.call_count, 1)
        args, kwargs = dev_label_transform.call_args
        self.assertEqual(args, ('OTHER',))
        self.assertEqual(kwargs, {})

        # test the test split
        test_label_transform = Mock(return_value='foo')
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=test_label_transform)
        # call the transform and check the return value
        _, _, label = test[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_label_transform.call_count, 1)
        args, kwargs = test_label_transform.call_args
        self.assertEqual(args, ('OTHER',))
        self.assertEqual(kwargs, {})


class ScruplesBenchmarkTestCase(unittest.TestCase):
    """Test scruples.dataset.readers.ScruplesBenchmark."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # copy the benchmark-easy dataset to the temporary directory
        for split in readers.ScruplesBenchmark.SPLITS:
            split_filename =\
                scruples_settings.BENCHMARK_FILENAME_TEMPLATE.format(
                    split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(settings.BENCHMARK_EASY_DIR, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_benchmark_has_correct_splits(self):
        self.assertEqual(
            set(readers.ScruplesBenchmark.SPLITS),
            set(['train', 'dev', 'test']))

    def test_reads_in_splits(self):
        # read the dataset
        benchmark = readers.ScruplesBenchmark(data_dir=self.temp_dir.name)

        # train
        train_ids, train_features, train_labels = benchmark.train
        self.assertIsInstance(train_ids, pd.Series)
        self.assertEqual(train_ids.tolist()[0], 'id_0')
        self.assertIsInstance(train_features, pd.DataFrame)
        self.assertEqual(
            train_features.to_dict(orient='records')[0],
            {
                "action0": "A good action.",
                "action1": "A bad action."
            })
        self.assertIsInstance(train_labels, pd.Series)
        self.assertEqual(train_labels.tolist()[0], 0)

        # dev
        dev_ids, dev_features, dev_labels = benchmark.dev
        self.assertIsInstance(dev_ids, pd.Series)
        self.assertEqual(dev_ids.tolist()[0], 'id_16')
        self.assertIsInstance(dev_features, pd.DataFrame)
        self.assertEqual(
            dev_features.to_dict(orient='records')[0],
            {
                "action0": "The good action.",
                "action1": "The bad action."
            })
        self.assertIsInstance(dev_labels, pd.Series)
        self.assertEqual(dev_labels.tolist()[0], 0)

        # test
        test_ids, test_features, test_labels = benchmark.test
        self.assertIsInstance(test_ids, pd.Series)
        self.assertEqual(test_ids.tolist()[0], 'id_20')
        self.assertIsInstance(test_features, pd.DataFrame)
        self.assertEqual(
            test_features.to_dict(orient='records')[0],
            {
                "action0": "Indeed, this describes a very good action.",
                "action1": "Indeed, this describes a very bad action."
            })
        self.assertIsInstance(test_labels, pd.Series)
        self.assertEqual(test_labels.tolist()[0], 0)


class ScruplesBenchmarkDatasetTestCase(unittest.TestCase):
    """Test scruples.dataset.readers.ScruplesBenchmarkDataset."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # copy the benchmark-easy dataset to the temporary directory
        for split in readers.ScruplesBenchmarkDataset.SPLITS:
            split_filename =\
                scruples_settings.BENCHMARK_FILENAME_TEMPLATE.format(
                    split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(settings.BENCHMARK_EASY_DIR, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_benchmark_has_correct_splits(self):
        self.assertEqual(
            set(readers.ScruplesBenchmarkDataset.SPLITS),
            set(['train', 'dev', 'test']))

    def test_init_raises_error_if_bad_split_provided(self):
        with self.assertRaisesRegex(
                ValueError,
                r'split must be one of train, dev, test\.'
        ):
          readers.ScruplesBenchmarkDataset(
              data_dir=self.temp_dir.name,
              split='val',
              transform=None,
              label_transform=None)

    def test_len(self):
        # test len on train
        train = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None)
        self.assertEqual(len(train), 16)

        # test len on dev
        dev = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None)
        self.assertEqual(len(dev), 4)

        # test len on test
        test = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None)
        self.assertEqual(len(test), 4)

    def test___get_item__(self):
        # test __get_item__ on train
        train = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None)
        id_, feature, label = train[0]
        self.assertEqual(id_, 'id_0')
        self.assertEqual(feature, (
            "A good action.",
            "A bad action."
        ))
        self.assertEqual(label, 0)

        # test __get_item__ on dev
        dev = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None)
        id_, feature, label = dev[0]
        self.assertEqual(id_, 'id_16')
        self.assertEqual(feature, (
            "The good action.",
            "The bad action."
        ))
        self.assertEqual(label, 0)

        # test __get_item__ on test
        test = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None)
        id_, feature, label = test[0]
        self.assertEqual(id_, 'id_20')
        self.assertEqual(feature, (
            "Indeed, this describes a very good action.",
            "Indeed, this describes a very bad action."
        ))
        self.assertEqual(label, 0)

    def test_init_with_transform(self):
        # test the train split
        train_transform = Mock(return_value='foo')
        train = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=train_transform,
            label_transform=None)
        # call the transform and check the return value
        _, feature, _ = train[0]
        self.assertEqual(feature, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_transform.call_count, 1)
        args, kwargs = train_transform.call_args
        self.assertEqual(args, ((
            "A good action.",
            "A bad action."
        ),))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_transform = Mock(return_value='foo')
        dev = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=dev_transform,
            label_transform=None)
        # call the transform and check the return value
        _, feature, _ = dev[0]
        self.assertEqual(feature, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_transform.call_count, 1)
        args, kwargs = dev_transform.call_args
        self.assertEqual(args, ((
            "The good action.",
            "The bad action."
        ),))
        self.assertEqual(kwargs, {})

        # test the test split
        test_transform = Mock(return_value='foo')
        test = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=test_transform,
            label_transform=None)
        # call the transform and check the return value
        _, feature, _ = test[0]
        self.assertEqual(feature, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_transform.call_count, 1)
        args, kwargs = test_transform.call_args
        self.assertEqual(args, ((
            "Indeed, this describes a very good action.",
            "Indeed, this describes a very bad action."
        ),))
        self.assertEqual(kwargs, {})

    def test_init_with_label_transform(self):
        # test the train split
        train_label_transform = Mock(return_value='foo')
        train = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=train_label_transform)
        # call the transform and check the return value
        _, _, label = train[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_label_transform.call_count, 1)
        args, kwargs = train_label_transform.call_args
        self.assertEqual(args, (0,))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_label_transform = Mock(return_value='foo')
        dev = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=dev_label_transform)
        # call the transform and check the return value
        _, _, label = dev[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_label_transform.call_count, 1)
        args, kwargs = dev_label_transform.call_args
        self.assertEqual(args, (0,))
        self.assertEqual(kwargs, {})

        # test the test split
        test_label_transform = Mock(return_value='foo')
        test = readers.ScruplesBenchmarkDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=test_label_transform)
        # call the transform and check the return value
        _, _, label = test[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_label_transform.call_count, 1)
        args, kwargs = test_label_transform.call_args
        self.assertEqual(args, (0,))
        self.assertEqual(kwargs, {})
