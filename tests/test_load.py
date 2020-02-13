import unittest
import numpy as np
from pathlib import Path
from undt.load import load_single, merge_data, merge_load, train_val_test_split
from sklearn.model_selection import train_test_split

def fun(x):
    return x + 1

class TestLoad(unittest.TestCase):

    def test_load_single(self):
        fpath = Path.cwd().parent / 'data/test_data.npz'
        data = load_single(fpath)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0].ndims, 3)
        self.assertEqual(data[1].ndims, 2)

    def test_merge_data(self):
        A, B, C = np.random.rand(3,1), np.random.rand(2,1), np.random.rand(1,1)
        data = [[A, B, C]] * 5
        A2, B2, C2 = np.vstack([A]*5), np.vstack([B]*5), np.vstack([C]*5)
        self.assertEqual(merge_data(data), [A2, B2, C2])

    def test_merge_load(self):
        fpath = Path.cwd().parent / 'data/test_data.npz'
        data = load_single(fpath)
        merged_data = merge_load([fpath] * 2)
        for d_s, d_m  in zip(data, merged_data):
            m_split = np.array_split(d_m, 2)
            self.assertEqual(d_s, m_split)

    def test_train_val_test_split(self):
        fpath = Path.cwd().parent / 'data/test_data.npz'
        data = load_single(fpath)
        tvt_data = train_val_test_split(data, split=[0.5, 0.5], random_state=1)
        tt_data = train_test_split(*data, train_size=0.5, random_state=1)
        self.assertEqual(tvt_data[2::3], tt_data[1::2])



if __name__ == '__main__':
    unittest.main()