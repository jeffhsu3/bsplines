"""
Unit tests for the b-spline module.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import bsplines as bs

from numpy.testing import (
        TestCase, run_module_suite, assert_equal, assert_,
        assert_almost_equal, assert_raises, assert_equal, dec)


def test_bspline_values():
    # Test computed b-spline values against Bernstein polynomials.
    numval = 10
    maxord = 10

    x = np.linspace(0, 1, numval)
    y = 1 - x
    b = np.ones((1, numval))
    for k in range(maxord):
        # Check k + 1 degree k b-splines.
        n = k + 1
        t = [0]*n + [1]*n
        c = np.eye(n)
        for j in range(n):
            msg = "k = %d, j = %d" % (k, j)
            tck = bs.Tck(t, c[j], k)
            res = bs.bsplval(x, tck)
            assert_almost_equal(res, b[j], err_msg=msg)
        # Prepare next set of Bernstein polynomial values if needed.
        if n < maxord:
            tmp = b
            b = np.zeros((n + 1, numval))
            b[:n] += y*tmp
            b[1:] += x*tmp


def test_bspvander():
    numval = 10
    maxord = 10

    x = np.linspace(0, 10, numval)
    for k in range(maxord):
        # Check k + 1 degree k b-splines.
        n = k + 1
        t = [0]*n + list(range(1, 10)) + [10]*n
        c = np.eye(len(t) - n)
        v = bs.bsplvander(x, t, k)
        for j in range(n):
            msg = "k = %d, j = %d" % (k, j)
            tck = bs.Tck(t, c[j], k)
            tgt = bs.bsplval(x, tck)
            assert_almost_equal(v[:, j], tgt, err_msg=msg)

def test_bspderiv():
    pass
    numval = 10
    maxord = 10
    dx = 1e-6

    xl = np.linspace(0, 10, numval)
    xr = xl + dx
    xm = xl + dx/2
    for k in range(1, maxord):
        # Check k + 1 degree k b-splines.
        n = k + 1
        t = [0]*n + list(range(1, 10)) + [10]*n
        c = np.eye(len(t) - n)
        for j in range(n):
            msg = "k = %d, j = %d" % (k, j)
            tck = bs.Tck(t, c[j], k)
            dck = bs.bsplderiv(tck, n=1)
            tgt = (bs.bsplval(xr, tck) - bs.bsplval(xl, tck))/dx
            res = bs.bsplval(xm, dck)
            assert_almost_equal(res, tgt, err_msg=msg)


if __name__ == "__main__":
    run_module_suite()
