import unittest
import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path
from undt.load import load_single, merge_data, merge_load, train_val_test_split
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
        fpath = Path(__file__).parents[1] / 'data/test_data.npz'
        data = load_single(fpath)
        tvt_data = train_val_test_split(data, split=[0.5, 0.5], random_state=1)
        tt_data = train_test_split(*data, train_size=0.5, random_state=1)
        for i, j in zip(tvt_data[2::3], tt_data[1::2]):
            self.assertIsNone(assert_array_equal(i, j))



if __name__ == '__main__':
    unittest.main()