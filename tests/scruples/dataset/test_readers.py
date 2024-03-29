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
        train_ids, train_features, train_labels, train_label_scores =\
            corpus.train
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
        self.assertIsInstance(train_label_scores, pd.DataFrame)
        self.assertEqual(
            {
                k: v
                for k, v in zip(
                        train_label_scores.columns,
                        train_label_scores.to_records()[0].tolist()[1:])
            },
            {"AUTHOR": 0, "OTHER": 10, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0})

        # dev
        dev_ids, dev_features, dev_labels, dev_label_scores = corpus.dev
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
        self.assertIsInstance(dev_label_scores, pd.DataFrame)
        self.assertEqual(
            {
                k: v
                for k, v in zip(
                        dev_label_scores.columns,
                        dev_label_scores.to_records()[0].tolist()[1:])
            },
            {"AUTHOR": 0, "OTHER": 10, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0})

        # test
        test_ids, test_features, test_labels, test_label_scores = corpus.test
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
        self.assertIsInstance(test_label_scores, pd.DataFrame)
        self.assertEqual(
            {
                k: v
                for k, v in zip(
                        test_label_scores.columns,
                        test_label_scores.to_records()[0].tolist()[1:])
            },
            {"AUTHOR": 0, "OTHER": 10, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0})


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
              label_transform=None,
              label_scores_transform=None)

    def test_len(self):
        # test len on train
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        self.assertEqual(len(train), 20)

        # test len on dev
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        self.assertEqual(len(dev), 5)

        # test len on test
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        self.assertEqual(len(test), 5)

    def test___get_item__(self):
        # test __get_item__ on train
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        id_, feature, label, label_scores = train[0]
        self.assertEqual(id_, 'id_0')
        self.assertEqual(feature, (
            "AITA test post",
            "This post is nta."
        ))
        self.assertEqual(label, 'OTHER')
        self.assertEqual(
            label_scores,
            {"AUTHOR": 0, "OTHER": 10, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0})

        # test __get_item__ on dev
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None)
        id_, feature, label, label_scores = dev[0]
        self.assertEqual(id_, 'id_20')
        self.assertEqual(feature, (
            "AITA test post",
            "Label this post nta."
        ))
        self.assertEqual(label, 'OTHER')
        self.assertEqual(
            label_scores,
            {"AUTHOR": 0, "OTHER": 10, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0})

        # test __get_item__ on test
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None)
        id_, feature, label, label_scores = test[0]
        self.assertEqual(id_, 'id_25')
        self.assertEqual(feature, (
            "AITA test post",
            "The label for this post should be nta."
        ))
        self.assertEqual(label, 'OTHER')
        self.assertEqual(
            label_scores,
            {"AUTHOR": 0, "OTHER": 10, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0})

    def test_init_with_transform(self):
        # test the train split
        train_transform = Mock(return_value='foo')
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=train_transform,
            label_transform=None,
            label_scores_transform=None)
        # call the transform and check the return value
        _, feature, _, _ = train[0]
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
            label_transform=None,
            label_scores_transform=None)
        # call the transform and check the return value
        _, feature, _, _ = dev[0]
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
            label_transform=None,
            label_scores_transform=None)
        # call the transform and check the return value
        _, feature, _, _ = test[0]
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
            label_transform=train_label_transform,
            label_scores_transform=None)
        # call the transform and check the return value
        _, _, label, _ = train[0]
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
            label_transform=dev_label_transform,
            label_scores_transform=None)
        # call the transform and check the return value
        _, _, label, _ = dev[0]
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
            label_transform=test_label_transform,
            label_scores_transform=None)
        # call the transform and check the return value
        _, _, label, _ = test[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_label_transform.call_count, 1)
        args, kwargs = test_label_transform.call_args
        self.assertEqual(args, ('OTHER',))
        self.assertEqual(kwargs, {})

    def test_init_with_label_scores_transform(self):
        # test the train split
        train_label_scores_transform = Mock(return_value='foo')
        train = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=train_label_scores_transform)
        # call the transform and check the return value
        _, _, _, label_scores = train[0]
        self.assertEqual(label_scores, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_label_scores_transform.call_count, 1)
        args, kwargs = train_label_scores_transform.call_args
        self.assertEqual(
            args,
            (
                {
                    'AUTHOR': 0,
                    'OTHER': 10,
                    'EVERYBODY': 0,
                    'NOBODY': 0,
                    'INFO': 0
                },
            ))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_label_scores_transform = Mock(return_value='foo')
        dev = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None,
            label_scores_transform=dev_label_scores_transform)
        # call the transform and check the return value
        _, _, _, label_scores = dev[0]
        self.assertEqual(label_scores, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_label_scores_transform.call_count, 1)
        args, kwargs = dev_label_scores_transform.call_args
        self.assertEqual(
            args,
            (
                {
                    'AUTHOR': 0,
                    'OTHER': 10,
                    'EVERYBODY': 0,
                    'NOBODY': 0,
                    'INFO': 0
                },
            ))
        self.assertEqual(kwargs, {})

        # test the test split
        test_label_scores_transform = Mock(return_value='foo')
        test = readers.ScruplesCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None,
            label_scores_transform=test_label_scores_transform)
        # call the transform and check the return value
        _, _, _, label_scores = test[0]
        self.assertEqual(label_scores, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_label_scores_transform.call_count, 1)
        args, kwargs = test_label_scores_transform.call_args
        self.assertEqual(
            args,
            (
                {
                    'AUTHOR': 0,
                    'OTHER': 10,
                    'EVERYBODY': 0,
                    'NOBODY': 0,
                    'INFO': 0
                },
            ))
        self.assertEqual(kwargs, {})


class ScruplesResourceTestCase(unittest.TestCase):
    """Test scruples.dataset.readers.ScruplesResource."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # copy the resource-easy dataset to the temporary directory
        for split in readers.ScruplesResource.SPLITS:
            split_filename =\
                scruples_settings.RESOURCE_FILENAME_TEMPLATE.format(
                    split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(settings.RESOURCE_EASY_DIR, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resource_has_correct_splits(self):
        self.assertEqual(
            set(readers.ScruplesResource.SPLITS),
            set(['train', 'dev', 'test']))

    def test_reads_in_splits(self):
        # read the dataset
        resource = readers.ScruplesResource(data_dir=self.temp_dir.name)

        # train
        train_ids, train_features, train_labels, train_label_scores =\
            resource.train
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
        self.assertIsInstance(train_label_scores, pd.DataFrame)
        self.assertEqual(
            list(train_label_scores.to_records()[0])[1:],
            [1, 0])

        # dev
        dev_ids, dev_features, dev_labels, dev_label_scores =\
            resource.dev
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
        self.assertIsInstance(dev_label_scores, pd.DataFrame)
        self.assertEqual(
            list(dev_label_scores.to_records()[0])[1:],
            [1, 0])

        # test
        test_ids, test_features, test_labels, test_label_scores =\
            resource.test
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
        self.assertIsInstance(test_label_scores, pd.DataFrame)
        self.assertEqual(
            list(test_label_scores.to_records()[0])[1:],
            [1, 0])


class ScruplesResourceDatasetTestCase(unittest.TestCase):
    """Test scruples.dataset.readers.ScruplesResourceDataset."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # copy the resource-easy dataset to the temporary directory
        for split in readers.ScruplesResourceDataset.SPLITS:
            split_filename =\
                scruples_settings.RESOURCE_FILENAME_TEMPLATE.format(
                    split=split)
            utils.copy_pkg_resource_to_disk(
                pkg='tests',
                src=os.path.join(settings.RESOURCE_EASY_DIR, split_filename),
                dst=os.path.join(self.temp_dir.name, split_filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resource_has_correct_splits(self):
        self.assertEqual(
            set(readers.ScruplesResourceDataset.SPLITS),
            set(['train', 'dev', 'test']))

    def test_init_raises_error_if_bad_split_provided(self):
        with self.assertRaisesRegex(
                ValueError,
                r'split must be one of train, dev, test\.'
        ):
          readers.ScruplesResourceDataset(
              data_dir=self.temp_dir.name,
              split='val',
              transform=None,
              label_transform=None,
              label_scores_transform=None)

    def test_len(self):
        # test len on train
        train = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        self.assertEqual(len(train), 16)

        # test len on dev
        dev = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        self.assertEqual(len(dev), 4)

        # test len on test
        test = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        self.assertEqual(len(test), 4)

    def test___get_item__(self):
        # test __get_item__ on train
        train = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        id_, feature, label, label_scores = train[0]
        self.assertEqual(id_, 'id_0')
        self.assertEqual(feature, (
            "A good action.",
            "A bad action."
        ))
        self.assertEqual(label, 0)
        self.assertEqual(label_scores, [1, 0])

        # test __get_item__ on dev
        dev = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        id_, feature, label, label_scores = dev[0]
        self.assertEqual(id_, 'id_16')
        self.assertEqual(feature, (
            "The good action.",
            "The bad action."
        ))
        self.assertEqual(label, 0)
        self.assertEqual(label_scores, [1, 0])

        # test __get_item__ on test
        test = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None,
            label_scores_transform=None)
        id_, feature, label, label_scores = test[0]
        self.assertEqual(id_, 'id_20')
        self.assertEqual(feature, (
            "Indeed, this describes a very good action.",
            "Indeed, this describes a very bad action."
        ))
        self.assertEqual(label, 0)
        self.assertEqual(label_scores, [1, 0])

    def test_init_with_transform(self):
        # test the train split
        train_transform = Mock(return_value='foo')
        train = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=train_transform,
            label_transform=None,
            label_scores_transform=None)
        # call the transform and check the return value
        _, feature, _, _ = train[0]
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
        dev = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=dev_transform,
            label_transform=None,
            label_scores_transform=None)
        # call the transform and check the return value
        _, feature, _, _ = dev[0]
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
        test = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=test_transform,
            label_transform=None,
            label_scores_transform=None)
        # call the transform and check the return value
        _, feature, _, _ = test[0]
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
        train = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=train_label_transform,
            label_scores_transform=None)
        # call the transform and check the return value
        _, _, label, _ = train[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_label_transform.call_count, 1)
        args, kwargs = train_label_transform.call_args
        self.assertEqual(args, (0,))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_label_transform = Mock(return_value='foo')
        dev = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=dev_label_transform,
            label_scores_transform=None)
        # call the transform and check the return value
        _, _, label, _ = dev[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_label_transform.call_count, 1)
        args, kwargs = dev_label_transform.call_args
        self.assertEqual(args, (0,))
        self.assertEqual(kwargs, {})

        # test the test split
        test_label_transform = Mock(return_value='foo')
        test = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=test_label_transform,
            label_scores_transform=None)
        # call the transform and check the return value
        _, _, label, _ = test[0]
        self.assertEqual(label, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_label_transform.call_count, 1)
        args, kwargs = test_label_transform.call_args
        self.assertEqual(args, (0,))
        self.assertEqual(kwargs, {})

    def test_init_with_label_scores_transform(self):
        # test the train split
        train_label_scores_transform = Mock(return_value='foo')
        train = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None,
            label_scores_transform=train_label_scores_transform)
        # call the transform and check the return value
        _, _, _, label_scores = train[0]
        self.assertEqual(label_scores, 'foo')
        # check that the transform was called correctly
        self.assertEqual(train_label_scores_transform.call_count, 1)
        args, kwargs = train_label_scores_transform.call_args
        self.assertEqual(args, ([1, 0],))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_label_scores_transform = Mock(return_value='foo')
        dev = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None,
            label_scores_transform=dev_label_scores_transform)
        # call the transform and check the return value
        _, _, _, label_scores = dev[0]
        self.assertEqual(label_scores, 'foo')
        # check that the transform was called correctly
        self.assertEqual(dev_label_scores_transform.call_count, 1)
        args, kwargs = dev_label_scores_transform.call_args
        self.assertEqual(args, ([1, 0],))
        self.assertEqual(kwargs, {})

        # test the test split
        test_label_scores_transform = Mock(return_value='foo')
        test = readers.ScruplesResourceDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None,
            label_scores_transform=test_label_scores_transform)
        # call the transform and check the return value
        _, _, _, label_scores = test[0]
        self.assertEqual(label_scores, 'foo')
        # check that the transform was called correctly
        self.assertEqual(test_label_scores_transform.call_count, 1)
        args, kwargs = test_label_scores_transform.call_args
        self.assertEqual(args, ([1, 0],))
        self.assertEqual(kwargs, {})
