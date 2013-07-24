import collections
import numpy as np


type_msg = "unsupported operand type(s) for %s: '%s' and '%s'"

def nonneg_int(i):
    i_ = int(i)
    if i_ != i or i_ < 0:
        raise ValueError("not a non-negative integer: '%s'" % i)
    return i_

#
# Validators
#


def asvalid_base(t, k, dtype=np.double, copy=0):
    """Convert and validate b-spline knot points and degrees.

    Converts degrees to equal non-negative integers, knot points to not
    writeable array of dtype `dtype`, and checks that the dtype is
    acceptable and the and the knot points are compatible with the
    degrees.

    Parameters
    ----------
    t : array_like
        list of array of knot points in non-decreasing order. The domain
        of the spline is the polytope with sides ``t[k] <= x <= t[m - k
        - 1]`` where k is the corresponding degree. No side can have
        zero length.  consequently the number ``n`` of knots defining
        each side must satisfy ``n >= 2*k + 2``, where `k` is the
        corresponding degree.
    k : number, sequence
        Can be a non-negative integer if domain is 1-D, else a sequence
        of non-negative integers. For this application 1.0 will be
        converted to 1 but 1.5 will fail.
    c : array_like, optional
        Array of coefficients of length equal to ``n - k - 1``,
        where ``k`` is the degree of the spline and ``n`` is the
        number of knot points
    dtype : {np.double, np.single}, optional
        The dtype of the knot points and coefficients.
    copy : {False, True}, optional
        Whether or not to force a copy of the arrays. If True, then the
        knot points are also made not writeable.

    Returns
    -------
    valid_t : list
        A list of valid knot arrays of dtype `dtype`.
    valid_k : list
        A list of non-negative integers taken from `k`.
    valid_c : {ndarray, None}
        Valid coefficient array of the specified dtype or None.
    valid_dtype: dtype
        The dtype of the knot points and the base dtype of the
        coefficients.

    Raises
    ------
    ValueError

    """
    # validate dtype
    if dtype != np.double and dtype != np.single:
        raise ValueError("Unsupported dtype %s" % dtype)

    # validate degrees
    if not isinstance(k, collections.Iterable):
        k = [k]
    k = [nonneg_int(i) for i in k]

    # validate knot points
    try:
        try:
            t = np.array(t, dtype=dtype, ndmin=1, copy=copy)
            t.setflags(write=0)
            if t.ndim == 1 and len(t) != 0:
                t = [t]
            elif t.ndim == 2:
                t = list[t]
            else:
                raise ValueError()
        except:
            t = [np.array(t_, dtype=dtype, ndmin=1, copy=copy) for t_ in t]
    except:
        raise ValueError("Unable to convert knot points to arrays.")
    if len(t) != len(k):
        raise ValueError("Knot points and degrees have different dimensions")
    if any((t_[1:] < t_[:-1]).any() for t_ in t):
        raise ValueError("Knot points must be non-decreasing")
    if any([len(t_) < 2 * k_ + 2 for t_, k_ in zip(t, k)]):
        raise ValueError("Not enough knot points for degrees")
    if any([t_[k_] == t_[-(k_+ 1)] for t_, k_ in zip(t, k)]):
        raise ValueError("Knot point interior must have positive length")
    if copy:
        for t_ in t:
            t_.setflags(write=0)

    return t, k, dtype


def asvalid_coef(c, domain_shape, dtype=np.double, copy=0, contiguous=0):
    c = np.array(c, dtype=dtype, copy=copy)
    if not c.shape[:len(domain_shape)] == domain_shape:
        raise ValueError("Dimensions do not match domain")
    if contiguous:
        c = np.ascontiguousarray(c)
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
    x = _asvalid_c_array(x, dtype=dtype)
    t, k, dtype = asvalid_base(t, k, dtype=dtype)
    k = k[0]
    t = t[0]
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


def _bsplval(x, bsp, axis=0):
    """Cython implementation of bsplval.

    See bsplval for documentation. All arguments are assumed valid.

    Parameters
    ----------
    bsp : BSpline
        The b-spline of which to take the derivative.
    n   : int
        The number of derivatives to take.

    Returns
    -------
    y : ndarray
       The b-spline evaluated at the points `x`

    """
    if axis >= bsp.domain_ndim:
        raise ValueError("Axis is out of range.")
    x = _asvalid_c_array(x, bsp.dtype)
    t_, c_, k_ = bsp.tck
    t = t_.pop(axis)
    c = np.rollaxis(c_, axis)
    k = k_.pop(axis)
    nord = k + 1
    m = len(x)
    n = len(t) - nord

    # Find the indexes of the upper ends of the non-empty
    # intervals and clip them to the valid interval so that
    # the spline can be extrapolated if needed.
    u = np.searchsorted(t, x, side='right')
    u.clip(nord, n, out=u)

    val = np.empty((m,) + c.shape[1:], dtype=bsp.dtype)
    ci = np.empty((nord,) + c.shape[1:], dtype=bsp.dtype)
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

    val = np.rollaxis(val, axis)

    # Return BSpline if there are still variables to evaluate over.
    if len(k_) != 0:
        # Debug, want check = 0 for production
        return BSpline(t_, k_, val)
    return val



def _bsplderiv(bsp, n, axis=0):
    """Cython implementation of bsplderiv.

    All arguments are assumed valid.

    Parameters
    ----------
    bsp : BSpline
        The spline parameters of which to take the derivative.
    n   : int
        The number of derivatives to take.

    Returns
    -------
    derivative : BSpline instance

    """
    t_, c_, k_ = bsp.tck
    t = t_[axis]
    c = np.rollaxis(c_, axis)
    k = k_[axis]
    dtype = bsp.dtype

    while n > 0:
        c = k * (c[1:] - c[:-1]) / (t[k + 1: -1] - t[1: -(k + 1)])
        t = t[1:-1]
        k -= 1
        n -= 1

    t_[axis] = t
    c_ = np.rollaxis(c, 0, axis + 1)
    k_[axis] = k

    # Debug, want check = 0 for production
    return BSpline(t_, k_, c_, dtype=dtype)


#
# Public interface
#

class BSDomain(object):

    def __init__(self, knots, degrees, dtype=np.double, check=1):
        if check:
            t, k, d = asvalid_base(knots, degrees, dtype=dtype, copy=0)
        self.__knots = t
        self.__degrees = k
        self.__dtype = d
        self.__ndim = len(k)
        self.__domain = [(a[n], a[-(n + 1)]) for n, a in zip(k, t)]
        self.__domain_shape = tuple([len(a) - n - 1 for n, a in zip(k, t)])


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.dtype != other.dtype:
            return False
        if self.degrees != other.degrees:
            return False
        if self.domain_shape != other.domain_shape:
            return False
        if any([(a != b).any() for a, b in zip(self.knots, other.knots)]):
            return False
        return True


    def __ne__(self, other):
        return not self.__eq__(other)


    def __copy__(self):
        # modified shallow copy
        return BSDomain(self.knots, self.degrees, self.dtype, check=0)


    @property
    def dtype(self):
        return self.__dtype


    @property
    def knots(self):
        return self.__knots[:]


    @property
    def degrees(self):
        return self.__degrees[:]


    @property
    def domain(self):
        return self.__domain[:]


    @property
    def domain_ndim(self):
        return self.__ndim


    @property
    def domain_shape(self):
        return self.__domain_shape


    def bspline(self, c):
        c = asvalid_coef(c, self.domain_shape, dtype=self.dtype, copy=1)
        #Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c, self.dtype)


class BSpline(BSDomain):
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


    def __init__(self, knots, degrees, coef, dtype=np.double, check=1):
        BSDomain.__init__(self, knots, degrees, check=check)
        self.__coef = asvalid_coef(coef, self.domain_shape, copy=1)


    def __eq__(self, other):
        return NotImplemented


    def __ne__(self, other):
        return NotImplemented


    def __copy__(self):
        # modified shallow copy
        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, self.coef, self.dtype)


    def __add__(self, other):
        c1 = self.coef
        if isinstance(other, self.__class__):
            if not self.eq_base(other):
                raise ValueError("Incompatible bases")
            c2 = other.coef
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
            raise ValueError("Incompatible array scalar")

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def __sub__(self, other):
        c1 = self.coef
        if isinstance(other, self.__class__):
            if not self.eq_base(other):
                raise ValueError("Incompatible bases")
            c2 = other.coef
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

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def __radd__(self, other):
        c1 = self.coef
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

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def __rsub__(self, other):
        c1 = self.coef
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

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def __mul__(self, other):
        if isinstance(other, self.__class__):
            msg = type_msg % (self.__class__.__name__, other)
            raise TypeError(msg)

        c1 = self.coef
        try:
            c2 = np.array(other, dtype=self.dtype)
        except:
            return NotImplemented

        # We only allow multiplication by "array scalars"
        if c2.ndim > self.range_ndim:
            raise ValueError("Incompatible scalar")

        try:
            c = c1 * c2
        except:
            raise ValueError("Incompatible scalar")

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def __div__(self, other):
        if isinstance(other, BSpline):
            msg = type_msg % (self.__class__.__name__, other)
            raise TypeError(msg)

        c1 = self.coef
        try:
            c2 = np.array(other, dtype=self.dtype)
        except:
            return NotImplemented

        # We only allow division by "array scalars"
        if c2.ndim > self.range_ndim:
            raise ValueError("Incompatible scalar")

        try:
            c = c1 / c2
        except:
            raise ValueError("Incompatible scalar")

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def __rmul__(self, other):
        c1 = self.coef
        try:
            c2 = np.array(other, dtype=self.dtype)
        except:
            raise ValueError("Scalar is not array_like")

        if c2.ndim > self.range_ndim:
            raise ValueError("Incompatible scalar")

        try:
            c = c2 * c1
        except:
            raise ValueError("Incompatible scalar")

        # Debug, want check = 0 for production
        return BSpline(self.knots, self.degrees, c)


    def eq_base(self, other):
        return BSDomain.__eq__(self, other)


    def get_base(self):
        return BSDomain(self.knots, self.degrees, self.dtype, check=0)


    @property
    def coef(self):
        return self.__coef.view()


    @property
    def range_ndim(self):
        return self.coef.ndim - self.domain_ndim


    @property
    def range_shape(self):
        return self.coef.shape[self.domain_ndim:]


    @property
    def tck(self):
        return self.knots, self.coef, self.degrees


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
    return _bsplvander(x, t, k, dtype=dtype)


def bsplval(x, bsp, axis=0):
    """Evaluate b-spline defined by t,c,k at x.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the spline.
    bsp : BSpline
        Instance of BSpline.

    Returns
    -------
    y : ndarray
        The b-spline evaluated at the points `x`. It has the type
        specified in `bsp`.

    """
    return _bsplval(x, bsp, axis=axis)


def bsplderiv(bsp, n=1, axis=0):
    """Take derivatives of the b-spline defined by bsp.

    All arguments are assumed valid.

    Parameters
    ----------
    bsp : BSpline
        The b-spline of which to take the derivative.
    n   : {1, int}
        Number of derivatives to take.

    Returns
    -------
    deriv : BSpline

    """
    n = nonneg_int(n)
    if not isinstance(bsp, BSpline):
        raise ValueError("bsp must be an instance of BSpline")
    if n > bsp.degrees[axis]:
        raise ValueError("n is larger than the spline degree")
    return _bsplderiv(bsp, n, axis)


def bsplinteg(bsp, n=1):
    pass


def bspzeros(bsp):
    pass


def bsplinterp(x, y):
    pass


def bsplcubic(x, y, mode='notaknot', dtype=np.double):
    pass

