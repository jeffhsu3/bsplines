import numpy as np


#
# Validators
#

class ConversionError(Exception):
    pass


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
    if np.issubdtype(dt, np.single):
        return dt
    else:
       raise ValueError("Unsupported dtype %s" % dt)


def _asvalid_k(k):
    """Return valid value of k if possible.

    For b-splines `k` must integer and >= 0

    Parameter
    ---------
    k : array_like
      The b-spline degrees.

    Returns
    -------
    valid_k : tuple
       If any element of `k` is equal to a non-negative integer, returns
       that value.

    Raises
    ------
    ValueError

    """
    k = np.array(k, ndmin=1)
    k_ = k.astype(np.int)
    if ((k_ != k) | (k_ < 0)).any():
        raise ValueError("Degrees must be non-negative integers.")
    return list(k_)


def _asvalid_t(t, k, dtype):
    """Return valid value of t and k if possible.

    Convert and validate the knot points `t`, given validated `k`
    and `dtype`.

    Parameter
    ---------
    t : list
       list of array of knot points in non-decreasing order. The domain of
       the spline is the polytope with sides ``t[k] <= x <= t[m - k - 1]``
       where k is the corresponding degree. No side can have zero length.
       consequently the number ``n`` of knots defining each side must satisfy
       ``n >= 2*k + 2``, where `k` is the corresponding degree.
    k : list,
       List of valid degrees.
    dtype : {single, double}
       Valid dtype for the knot points.

    Returns
    -------

    valid_t : list
       list of valid knot arrays of the specified dtype.

    Raises
    ------
    ValueError

    """
    try:
        try:
            t = np.array(t, dtype=dtype, ndmin=1)
            if t.ndim == 1 and len(t) != 0:
                t = [t]
            elif t.ndim == 2:
                t = [t_ for t_ in t]
            else:
                raise ValueError()
        except:
            t = [np.array(t_, dtype=dtype, ndmin=1) for t_ in t]
    except:
        raise ValueError("Unable to convert knot points to arrays.")
    if len(t) != len(k):
        raise ValueError("Knot points and degrees have different dimensions")
    if any((t_[1:] < t_[:-1]).any() for t_ in t):
        raise ValueError("Knot points must be non-decreasing")
    if any([len(t_) < 2 * k_ + 2 for t_, k_ in zip(t, k)]):
        raise ValueError("Not enough knot points for degrees")
    if any([t_[k_] == t_[k_+ 1] for t_, k_ in zip(t, k)]):
        raise ValueError("First interior interval must have positive length.")
    if any([t_[-(k_ + 2)] == t_[-(k_ + 1)] for t_, k_ in zip(t, k)]):
        raise ValueError("Last interior interval must have positive length.")
    return t


def _asvalid_c(c, t, k, dtype=np.double):
    """Return valid value of t, c, and k if possible.

    Convert and validate the coefficients `c` given validated `t`,
    `k`, and `dtype`.

    Parameter
    ---------
    c : array_like,
       Array of coefficients of length equal to ``n - k - 1``,
       where ``k`` is the degree of the spline and ``n`` is the
       number of knot points
    t : ndarray, shape (n,)
       Array of knot points, assumed validated.
    k : int
       Degree of the spline, assumed validated.

    Returns
    -------

    valid_c : ndarray
       Valid coefficient array of the specified dtype.

    Raises
    ------
    ValueError

    """
    c = np.array(c, dtype=dtype)
    if c.ndim < len(k):
        raise ValueError("To few dimensions for dimensionality.")
    if any([c.shape[i] != len(t[i]) - k[i] - 1 for i in range(len(k))]):
        raise ValueError("Wrong number of coefficients.")
    return c


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
    dtype : {single, double}
       The dtype of the knot points. Only np.double is currently
       supported.

    """
    __array_priority__ = 1000

    def __init__(self, t, c, k, dtype=np.double, ndim=1):
        k = _asvalid_k(k)
        t = _asvalid_t(t, k, dtype=dtype)
        c = _asvalid_c(c, t, k, dtype=dtype)
        self._k = k
        self._t = t
        self._c = c
        self._dom = [(a[n], a[-(n + 1)]) for n, a in zip(k, t)]
        self._dtype = dtype


    def __add__(self, other):
        c1 = self.c
        if isinstance(other, Tck):
            if not self._is_compatible_tck(other):
                raise ValueError("Incompatible knots")
            c2 = other.c
            n = c1.ndim - c2.ndim
            m = self.domain_ndim
            # Make vectors broadcast
            if n > 0:
                c2.shape = c2.shape[:m] + (1,)*n + c2.shape[m:]
            elif n < 0:
                c1.shape = c1.shape[:m] + (1,)*n + c1.shape[m:]
        else:
            try:
                c2 = np.array(other, dtype=self.dtype)
            except:
                return NotImplemented
            if c2.ndim > self.range_ndim:
                raise ValueError("Incompatible scalar")
        try:
            c = c1 + c2
        except:
            raise ValueError("Incompatible scalar")

        return Tck(self.t, c, self.k)


    def __sub__(self, other):
        c1 = self.c
        if isinstance(other, Tck):
            if not self._is_compatible_tck(other):
                raise ValueError("Incompatible knots")
            c2 = other.c
            n = c1.ndim - c2.ndim
            m = self.domain_ndim
            # Make vectors broadcast
            if n > 0:
                c2.shape = c2.shape[:m] + (1,)*n + c2.shape[m:]
            elif n < 0:
                c1.shape = c1.shape[:m] + (1,)*n + c1.shape[m:]
        else:
            try:
                c2 = np.array(other, dtype=self.dtype)
            except:
                return NotImplemented
            if c2.ndim > self.range_ndim:
                raise ValueError("Incompatible scalar")
        try:
            c = c1 - c2
        except:
            raise ValueError("Incompatible scalar")

        return Tck(self.t, c, self.k)


    def __radd__(self, other):
        c1 = self.c
        try:
            c2 = np.array(other, dtype=self.dtype)
        except:
            return NotImplemented
        if c2.ndim > self.range_ndim:
            raise ValueError("Incompatible scalar")

        try:
            c = c2 + c1
        except:
            raise ValueError("Incompatible scalar")

        return Tck(self.t, c, self.k)


    def __rsub__(self, other):
        c1 = self.c
        try:
            c2 = np.array(other, dtype=self.dtype)
        except:
            return NotImplemented
        if c2.ndim > self.range_ndim:
            raise ValueError("Incompatible scalar")

        try:
            c = c2 - c1
        except:
            raise ValueError("Incompatible scalar")

        return Tck(self.t, c, self.k)


    def _is_compatible_tck(self, other):
        if self.dtype != other.dtype:
            return False
        elif any([(t1 != t2).any() for t1, t2 in zip(self.t, other.t)]):
            return False
        else:
            return True


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


    @property
    def domain(self):
        return self._dom


    @property
    def domain_ndim(self):
        return len(self.k)


    @property
    def domain_shape(self):
        return self.c.shape[:len(self.k)]


    @property
    def range_ndim(self):
        return self.c.ndim - len(self.k)


    @property
    def range_shape(self):
        return self.c.shape[len(self.k):]


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

