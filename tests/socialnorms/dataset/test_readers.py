"""Tests for socialnorms.dataset.readers."""

import os
import pkg_resources
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

import socialnorms.settings as socialnorms_settings
from socialnorms.dataset import readers
from ... import settings


class SocialnormsCorpusTestCase(unittest.TestCase):
    """Test socialnorms.dataset.readers.SocialnormsCorpus."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # copy the socialnorms-easy dataset to the temporary
        # directory

        # train
        with pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_TRAIN_PATH
        ) as train_in,\
        open(
            os.path.join(
                self.temp_dir.name,
                socialnorms_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split='train')),
            'wb'
        ) as train_out:
            train_out.write(train_in.read())

        # dev
        with pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_DEV_PATH
        ) as dev_in,\
        open(
            os.path.join(
                self.temp_dir.name,
                socialnorms_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split='dev')),
            'wb'
        ) as dev_out:
            dev_out.write(dev_in.read())

        # test
        with pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_TEST_PATH
        ) as test_in,\
        open(
            os.path.join(
                self.temp_dir.name,
                socialnorms_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split='test')),
            'wb'
        ) as test_out:
            test_out.write(test_in.read())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_socialnorms_has_correct_splits(self):
        self.assertEqual(
            set(readers.SocialnormsCorpus.SPLITS),
            set(['train', 'dev', 'test']))

    def test_reads_in_splits(self):
        # read the dataset
        socialnorms = readers.SocialnormsCorpus(data_dir=self.temp_dir.name)

        # train
        train_ids, train_features, train_labels = socialnorms.train
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
        self.assertEqual(train_labels.tolist()[0], 'NTA')

        # dev
        dev_ids, dev_features, dev_labels = socialnorms.dev
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
        self.assertEqual(dev_labels.tolist()[0], 'NTA')

        # test
        test_ids, test_features, test_labels = socialnorms.test
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
        self.assertEqual(test_labels.tolist()[0], 'NTA')


class SocialnormsCorpusDatasetTestCase(unittest.TestCase):
    """Test socialnorms.dataset.readers.SocialnormsCorpusDataset."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # copy the socialnorms-easy dataset to the temporary
        # directory

        # train
        with pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_TRAIN_PATH
        ) as train_in,\
        open(
            os.path.join(
                self.temp_dir.name,
                socialnorms_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split='train')),
            'wb'
        ) as train_out:
            train_out.write(train_in.read())

        # dev
        with pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_DEV_PATH
        ) as dev_in,\
        open(
            os.path.join(
                self.temp_dir.name,
                socialnorms_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split='dev')),
            'wb'
        ) as dev_out:
            dev_out.write(dev_in.read())

        # test
        with pkg_resources.resource_stream(
                'tests', settings.SOCIALNORMS_EASY_TEST_PATH
        ) as test_in,\
        open(
            os.path.join(
                self.temp_dir.name,
                socialnorms_settings.CORPUS_FILENAME_TEMPLATE.format(
                    split='test')),
            'wb'
        ) as test_out:
            test_out.write(test_in.read())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_socialnormsdataset_has_correct_splits(self):
        self.assertEqual(
            set(readers.SocialnormsCorpusDataset.SPLITS),
            set(['train', 'dev', 'test']))

    def test_init_raises_error_if_bad_split_provided(self):
        with self.assertRaisesRegex(
                ValueError,
                r'split must be one of train, dev, test\.'
        ):
          readers.SocialnormsCorpusDataset(
              data_dir=self.temp_dir.name,
              split='val',
              transform=None,
              label_transform=None)

    def test_len(self):
        # test len on train
        train = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=None)
        self.assertEqual(len(train), 20)

        # test len on dev
        dev = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=None)
        self.assertEqual(len(dev), 5)

        # test len on test
        test = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=None)
        self.assertEqual(len(test), 5)

    def test___get_item__(self):
        # test __get_item__ on train
        train = readers.SocialnormsCorpusDataset(
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
        self.assertEqual(label, 'NTA')

        # test __get_item__ on dev
        dev = readers.SocialnormsCorpusDataset(
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
        self.assertEqual(label, 'NTA')

        # test __get_item__ on test
        test = readers.SocialnormsCorpusDataset(
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
        self.assertEqual(label, 'NTA')

    def test_init_with_transform(self):
        # test the train split
        train_transform = Mock()
        train = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=train_transform,
            label_transform=None)
        # get the item to call the transform
        train[0]
        self.assertEqual(train_transform.call_count, 1)
        args, kwargs = train_transform.call_args
        self.assertEqual(args, ((
            "AITA test post",
            "This post is nta."
        ),))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_transform = Mock()
        dev = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=dev_transform,
            label_transform=None)
        # get the item to call the transform
        dev[0]
        self.assertEqual(dev_transform.call_count, 1)
        args, kwargs = dev_transform.call_args
        self.assertEqual(args, ((
            "AITA test post",
            "Label this post nta."
        ),))
        self.assertEqual(kwargs, {})

        # test the test split
        test_transform = Mock()
        test = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=test_transform,
            label_transform=None)
        # get the item to call the transform
        test[0]
        self.assertEqual(test_transform.call_count, 1)
        args, kwargs = test_transform.call_args
        self.assertEqual(args, ((
            "AITA test post",
            "The label for this post should be nta."
        ),))
        self.assertEqual(kwargs, {})

    def test_init_with_label_transform(self):
        # test the train split
        train_label_transform = Mock()
        train = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='train',
            transform=None,
            label_transform=train_label_transform)
        # get the item to call the transform
        train[0]
        self.assertEqual(train_label_transform.call_count, 1)
        args, kwargs = train_label_transform.call_args
        self.assertEqual(args, ('NTA',))
        self.assertEqual(kwargs, {})

        # test the dev split
        dev_label_transform = Mock()
        dev = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='dev',
            transform=None,
            label_transform=dev_label_transform)
        # get the item to call the transform
        dev[0]
        self.assertEqual(dev_label_transform.call_count, 1)
        args, kwargs = dev_label_transform.call_args
        self.assertEqual(args, ('NTA',))
        self.assertEqual(kwargs, {})

        # test the test split
        test_label_transform = Mock()
        test = readers.SocialnormsCorpusDataset(
            data_dir=self.temp_dir.name,
            split='test',
            transform=None,
            label_transform=test_label_transform)
        # get the item to call the transform
        test[0]
        self.assertEqual(test_label_transform.call_count, 1)
        args, kwargs = test_label_transform.call_args
        self.assertEqual(args, ('NTA',))
        self.assertEqual(kwargs, {})
