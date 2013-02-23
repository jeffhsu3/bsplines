import numpy as np


#
# Validators
#

def _asvalid_dtype(dt):
    """Check supported dtype.

    Parameters
    ----------
    dt : dtype
        The dtype to be checked.

    Returns
    -------
    valid_dtype: dtype
        Same as `dt` if it is supported.

    Raises
    ------
    ValueError

    """
    if np.issubdtype(dt, np.double):
        return dt
    # np.single is not supported yet.
    if False and np.issubdtype(dt, np.single):
        return dt
    else:
       raise ValueError("Unsupported dtype %s" % dt)


def _asvalid_k(k):
    """Return valid value of k if possible.

    For b-splines `k` must integer and >= 0

    Parameter
    ---------
    k : int_like
      Value to be converted.

    Returns
    -------
    valid_k : int
       If `k` is equal to a non-negative integer, returns
       that value.

    Raises
    ------
    ValueError

    """
    k_ = int(k)
    if k_ != k or k_ < 0:
        raise ValueError("k must be non-negative integer.")
    return k_


def _asvalid_t(t, k, dtype=np.double, copy=False):
    """Return valid value of t and k if possible.

    Convert and validate the knot points `t`, given validated `k`
    and `dtype`.

    Parameter
    ---------
    t : array_like, shape (n,)
       Array of knot points in non-decreasing order. The domain of
       the spline is the closed interval ``t[k] <= x <= t[m - k - 1]``
       and it must not have zero length. It follows that we must have
       ``n >= 2*k + 2``, where `k` is the degree of the spline.
    k : int
       Degree of the b-spline. It is assumed to have been checked.
    dtype : {double, dtype}, optional
       The dtype of the knot points. It is assumed to have been checked.
    copy : {False, True}
       If true, always return a copy of the knot points.

    Returns
    -------

    valid_t : ndarray
       Valid knots of the specified dtype.

    Raises
    ------
    ValueError

    """
    t_ = np.array(t, dtype=dtype, ndmin=1, copy=copy)
    if t_.ndim > 1:
        raise ValueError("The knot points must be 1-D.")
    if t_.size < 2*k + 2:
        raise ValueError("Not enough knot points.")
    if (t_.argsort(kind='mergesort') != np.arange(t_.size)).any():
        raise ValueError("Knot points must be non-decreasing")
    return t_


def _asvalid_c(c, t, k, dtype=np.double, copy=False):
    """Return valid value of t, c, and k if possible.

    Convert and validate the coefficients `c` given validated `t`,
    `k`, and `dtype`.

    Parameter
    ---------
    c : array_like, shape (m, )
       Array of coefficients of length equal to ``n - k - 1``,
       where ``k`` is the degree of the spline and ``n`` is the
       number of knot points
    t : ndarray, shape (n,)
       Array of knot points, assumed to have been validated.
    k : int
       Degree of the spline, assumed to have been validated.
    dtype : {double, dtype}, optional
       The dtype of the knot points. It is assumed to have been checked.
    copy : {False, True}
       If true, always return a copy of the coefficients.



    Returns
    -------

    valid_c : ndarray
       Valid coefficient array of the specified dtype.

    Raises
    ------
    ValueError

    """
    c_ = np.array(c, dtype=dtype, ndmin=1, copy=False)
    if c_.ndim > 1:
        raise ValueError("The coefficients must be 1-D.")
    if c_.size != t.size - k - 1:
        raise ValueError("Wrong number of coefficients.")
    return c_


def _asvalid_c_array(x, dtype=np.double, copy=False, ndmin=1, maxdim=1):
    x = np.array(x, dtype=dtype, copy=copy, ndmin=ndmin)
    if x.ndim > maxdim:
        raise ValueError("Maximum dimensions allowed = %d" % maxdim)
    if not x.flags.c_contiguous:
        x = np.array(x, copy=True)
    return x


#
# These functions should be Cythonized at some point.
#


def _bsplvander(x, t, k, dtype=np.double):
    """Cython implementation of bsplvander.

    See bsplvander for documentation. All arguments are assumed valid.

    Parameters
    ----------
    x : array_like, shape (m,)
        1-D array of points at which to evaluate the spline functions.
    t : array_like, shape(n,)
        1-D array of knot points in non-decreasing order.
    k : int
        Degree of the spline.
    dtype: {double, dtype}, optional
        Only np.double is currently supported.

    Returns
    -------
    van : ndarray, shape(m, n - k - 1)
        A pseudo-Vandermonde matrix for the b-splines of degree `k` on
        the knot sequence `knots`.

    """
    nord = k + 1
    m = len(x)
    n = len(t) - nord

    # Find the indexes of the upper ends of the non-empty
    # intervals and clip them to the valid interval so that
    # the spline can be extrapolated.
    u = np.searchsorted(t, x, side='right')
    u.clip(nord, n, out=u)


    van = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        ui = u[i]
        xi = x[i]
        ti = t[ui - nord:]
        row = van[i, ui - nord:]
        row[k] = 1.
        for j in range(1, nord):
            for l in range(j):
                ul = nord + l
                uj = ul - j
                tmp = row[uj]
                a = (xi - ti[uj]) / (ti[ul] - ti[uj])
                row[uj] = a * tmp
                row[uj - 1] += (1 - a) * tmp

    return van


def _bsplval(tck, x):
    """Cython implementation of bsplval.

    See bsplval for documentation. All arguments are assumed valid.

    Parameters
    ----------
    tck : Tck
        The b-spline of which to take the derivative.
    n   : int
        The number of derivatives to take.

    Returns
    -------
    y : ndarray
       The b-spline evaluated at the points `x`

    """
    t, c, k = tck.tck
    nord = k + 1
    m = len(x)
    n = len(t) - nord

    # Find the indexes of the upper ends of the non-empty
    # intervals and clip them to the valid interval so that
    # the spline can be extrapolated if needed.
    u = np.searchsorted(t, x, side='right')
    u.clip(nord, n, out=u)

    val = np.empty((m,), dtype=tck.dtype)
    ci = np.empty((nord,), dtype=tck.dtype)
    for i in range(m):
        ui = u[i]
        xi = x[i]
        ti = t[ui - k:]
        ci[...] = c[ui - nord: ui]
        for j in range(k):
            for l in range(k - j):
                a = (xi - ti[j + l]) / (ti[k + l] - ti[j + l])
                ci[l] = (1 - a) * ci[l] + a * ci[l + 1]
        val[i] = ci[0]

    return val


def _bsplderiv(tck, n):
    """Cython implementation of bsplderiv.

    All arguments are assumed valid.

    Parameters
    ----------
    tck : Tck
        The spline parameters of which to take the derivative.
    n   : int
        The number of derivatives to take.

    Returns
    -------
    derivative : Tck instance

    """
    t, c, k = tck.tck
    dtype = tck.dtype

    for i in range(n):
        c = k * (c[1:] - c[:-1]) / (t[k + 1: -1] - t[1: -(k + 1)])
        t = t[1:-1]
        k = k - 1

    return Tck(t, c, k, dtype=dtype)


#
# Public interface
#

class Tck(object):
    """Class to hold tck values for b-splines.

    This is needed so that we do not need to validate the contents
    whenever the knot points, coefficients, and degree is needed to
    evaluate a spline at the C level.

    Parameters
    ----------
    t : array_like, shape (m,)
       Array of knot points in non-decreasing order. The domain of
       the spline is the closed interval ``t[k] <= x <= t[m - k - 1]``
       and it must not have zero length. It follows that we must have
       ``n >= 2*k + 2``, where `k` is the degree of the spline.
    c : array_like, shape (n, )
       Array of coefficients. Its length must satisfy ``n = m - k - 1``,
       where `k` is the degree of the spline.
    k : int
       Degree of the spline, It must be >= 0.
    dtype : {double, dtype}, optional
       The dtype of the knot points. Only np.double is currently
       supported.

    """
    def __init__(self, t, c, k, dtype=None):
        # Try to keep float 32 if dtypes are available
        if dtype is None:
            types = [d.dtype for d in [c, t] if isinstance(d, np.ndarray)]
            dtype = np.find_common_type(types, [])
        if dtype is None:
            dtype = np.double
        dtype = _asvalid_dtype(dtype)
        k = _asvalid_k(k)
        t = _asvalid_t(t, k, dtype=dtype, copy=True)
        c = _asvalid_c(c, t, k, dtype=dtype, copy=True)
        self._dtype = dtype
        self._k = k
        self._t = t
        self._c = c

    @property
    def dtype(self):
        return self._dtype

    @property
    def t(self):
        return self._t

    @property
    def c(self):
        return self._c

    @property
    def k(self):
        return self._k

    @property
    def tck(self):
        return self._t, self._c, self._k


class BSpline(Tck):

    def __call__(self, x):
        """Evaluate b-spline at points x.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the spline.

        Returns
        -------
        y : ndarray
            The b-spline evaluated at the points `x`. It has the type
            specified by `dtype`.

        """
        x = _asvalid_c_array(x, dtype=dtype)
        return _bsplval(self, x)


    def deriv(self, n):
        n = asvalid_k(n)
        if n > self._k:
            raise ValueError("n is larger than the spline degree")
        return _bsplderiv(self, n)


def bsplvander(x, t, k, dtype=np.double):
    """Pseudo-Vandermonde matrix of b-spline basis functions.

    The returned matrix is defined by

        ``v[i, j] = N_{j, k}(x[i])``

    Where ``N_{j, k}`` is the j'th basis b-spline of degree `k`. The
    number of basis splines is ``n - k - 1``, where n is the number
    of knots.

    The knot points must be non-decreasing and the multiplicity of any
    given knot must be less than `k` + 1. Note that the knots in the
    closed interval of definition are assumed to be extended at both
    ends by `k` new knots. Usually those knots are repeats of the end
    points of the valid interval, but they need not be. Consequently the
    end points are ``t[k]`` and ``t[n - k - 1]``.

    The points of `x` can be any valid floating point values. If a value
    is located outside of the valid interval, then the b-splines that
    are non-zero on the closest subinterval in the valid interval are
    extrapolated as polynomials. This allows for roundoff error at the
    end points and the inclusion of the right end point in the valid
    interval.

    Parameters
    ----------
    x : array_like, shape (m,)
        1-D array of points at which to evaluate the spline functions.
    t : array_like, shape(n,)
        1-D array of knot points in non-decreasing order.
    k : int
        Degree of the spline.
    dtype: {double, dtype}, optional
        Only double is currently supported.

    Returns
    -------
    van : ndarray, shape (m, n - k - 1)
        Pseudo-Vandermonde matrix for the b-splines of degree `k` on the
        knot sequence `t`.

    Raises
    ------
    ValueError

    Examples
    --------

    >>> from bsplines import bsplvander
    >>> knots = [0]*4 + [1]*4
    >>> x = np.linspace(0, 1)
    >>> v = bsplvander(x, knots, 3)
    >>> v.shape
    (50, 4)

    """
    dtype = _asvalid_dtype(dtype)
    k = _asvalid_k(k)
    t = _asvalid_t(t, k, dtype=dtype)
    x = _asvalid_c_array(x, dtype=dtype)
    return _bsplvander(x, t, k, dtype=dtype)


def bsplval(x, tck):
    """Evaluate b-spline defined by t,c,k at x.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the spline.
    tck : Tck
        Instance of Tck.

    Returns
    -------
    y : ndarray
        The b-spline evaluated at the points `x`. It has the type
        specified in `tck`.

    """
    _asvalid_c_array(x, dtype=tck.dtype)
    return _bsplval(tck, x)


def bsplderiv(tck, n=1):
    """Take derivatives of the b-spline defined by tck.

    All arguments are assumed valid.

    Parameters
    ----------
    tck : Tck
        The b-spline of which to take the derivative.
    n   : {1, int}
        Number of derivatives to take.

    Returns
    -------
    deriv : Tck

    """
    n = _asvalid_k(n)
    if not isinstance(tck, Tck):
        raise ValueError("tck must be an instance of Tck")
    if n > tck.k:
        raise ValueError("n is larger than the spline degree")
    return _bsplderiv(tck, n)


def bsplinteg(tck, n=1):
    pass


def bspzeros(tck):
    pass


def bsplinterp(x, y):
    pass


def bsplcubic(x, y, mode='notaknot', dtype=np.double):
    pass

