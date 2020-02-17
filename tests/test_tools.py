import unittest
import numpy as np
from undt.tools import normal
from numpy.testing import assert_array_equal

class TestTools(unittest.TestCase):

    def test_normal(self):
        X = normal(np.random.rand(100, 20))
        Y = normal(X)
        self.assertIsNone(assert_array_equal(np.max(Y, axis=1), np.ones(100)))
