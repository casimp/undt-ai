import unittest
import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path
from undt.load import load_single, merge_data, merge_load, train_val_test_split
from undt.load import load_val_split, load_pipeline, merge_load_pipeline
from undt.load import merge_load_split_pipeline
from sklearn.model_selection import train_test_split
import os

class TestLoad(unittest.TestCase):

    def test_load_single(self):
        fpath = Path(__file__).parents[1] / 'data/test_data.npz'
        data = load_single(fpath)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0].ndim, 3)
        self.assertEqual(data[1].ndim, 2)

    def test_merge_data(self):
        A, B, C = np.random.rand(3,1), np.random.rand(2,1), np.random.rand(1,1)
        data = [[A, B, C]] * 5
        A2 = np.vstack([A]*5)
        self.assertIsNone(assert_array_equal(merge_data(data)[0], A2))

    def test_merge_load(self):
        fpath = Path(__file__).parents[1] / 'data/test_data.npz'
        data = load_single(fpath)
        merged_data = merge_load([fpath] * 2)
        for d_s, d_m  in zip(data, merged_data):
            m_split = np.array_split(d_m, 2, axis=0)[0]
            self.assertIsNone(assert_array_equal(d_s, m_split))

    def test_train_val_test_split(self):
        data = [np.random.rand(100, 1), np.random.rand(100, 1)]
        tvt_data = train_val_test_split(data, split=[0.5, 0.5], random_state=1)
        tt_data = train_test_split(*data, train_size=0.5, random_state=1)
        for i, j in zip(tvt_data[2::3], tt_data[1::2]):
            self.assertIsNone(assert_array_equal(i, j))

    def test_load_val_split(self):
        fpath = Path(__file__).parents[1] / 'data/test_data.npz'
        tvt_data = load_val_split(fpath, split=[0.5, 0.5], random_state=1)
        tt_data = load_single(fpath)
        tt_data = train_test_split(*tt_data, train_size=0.5, random_state=1)

        for i, j in zip(tvt_data[2::3], tt_data[1::2]):
            self.assertIsNone(assert_array_equal(i, j))

    def test_load_pipeline(self):
        fpath = Path(__file__).parents[1] / 'data/test_data.npz'
        data = load_val_split(fpath)
        pipe_data = load_pipeline(fpath, levels=[0, 0.01, 0.02, 0.04])
        self.assertEqual(pipe_data[0].size / 4, data[0].size)
        pipe_data = load_pipeline(fpath, levels=[0,])
        self.assertEqual(pipe_data[0].size, data[0].size)
        for i, j in zip(data, pipe_data):
            self.assertIsNone(assert_array_equal(i, j))

    def test_merge_load_split_pipeline(self):
        fpath = Path(__file__).parents[1] / 'data/test_data.npz'
        data = merge_load_pipeline([fpath, fpath], levels=[0,])
        merged_data = merge_load_split_pipeline([[fpath]*2, [fpath]*2], 
                                                 [[0,], [0.1,]])
        for d_s, d_m  in zip(data, merged_data):
            m_split = np.array_split(d_m, 2, axis=0)[0]
            self.assertIsNone(assert_array_equal(d_s, m_split))


if __name__ == '__main__':
    unittest.main()