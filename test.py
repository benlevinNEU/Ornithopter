import unittest
from LookUpTableGen import compute_alpha, dc

class TestComputeAlpha(unittest.TestCase):
    def test_compute_alpha_1(self):
        theta, si, beta, alpha = compute_alpha(327.02/dc, -54.40/dc)
        self.assertAlmostEqual(beta*dc%360, 207.46, delta=0.1)
        self.assertAlmostEqual(alpha*dc, 0, delta=2)

    def test_compute_alpha_2(self):
        theta, si, beta, alpha = compute_alpha(121.32/dc, -74.17/dc)
        self.assertAlmostEqual(beta*dc%360, 172.18, delta=0.1)
        self.assertAlmostEqual(alpha*dc, 0, delta=1)

    def test_compute_alpha_3(self):
        theta, si, beta, alpha = compute_alpha(257.44/dc, -100.45/dc)
        self.assertAlmostEqual(beta*dc%360, 208.18, delta=0.11)
        self.assertAlmostEqual(alpha*dc, 0, delta=1)

    '''def test_compute_alpha_4(self):
        theta, si, beta, alpha = compute_alpha(148.27/dc, -100.12/dc)
        self.assertAlmostEqual(beta*dc, -180, delta=1)
        self.assertAlmostEqual(alpha*dc, 0, delta=1)'''

if __name__ == '__main__':
    unittest.main()