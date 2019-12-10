import math

import numpy as np

from .. import AutoDiff, AutoDiffRev, AutoDiffVector, AutoDiffRevVector

def __update_unary(x, operation, doperation):
    """Updates an AutoDiff object with a unary operation or simply
        performs the unary operation on a numeric if it's supplied
        Not to be used externally.
    
    INPUTS
    =======
    x: The AutoDiff object or numeric. Uses ducktyping.
    operation: function, the unary operation, which is used to uodate the value 
    doperation: the derivative of the unary operation, which is used to update
                the gradients
    
    RETURNS
    ========
    An AutoDiff object whose value and gradient have been updated by the
     unary operation
    """

    #attempt to broadcast uperation to each element of iterable if possible

    if isinstance(x, AutoDiffVector):
        results = []
        for ad in x:
            results.append(__update_unary(ad, operation, doperation))

        return AutoDiffVector(results)

    # if isinstance(x, AutoDiffRevVector):

    #     results = []
    #     for ad in x:
    #         results.append(__update_unary(ad, operation, doperation))

    #     return AutoDiffRevVector(results)


    if isinstance(x, AutoDiffRev):
        sig = AutoDiffRev.generate_signature()
        updated_breadcrumbs = x.breadcrumbs | set([sig])    
        
        # keep track of root variables
        updated_root_vars = x.root_vars.copy()

        updated_names_init_vals = x.get_names_init_vals()

        val = x.get_value()

        updated_val = operation(val)

        if np.isnan(updated_val):
                raise ValueError

        weight = doperation(val)

        if np.isnan(weight):
            raise ValueError
        
        z = AutoDiffRev(val=updated_val,
                        breadcrumbs=updated_breadcrumbs,
                        root_vars=updated_root_vars,
                        names_init_vals=updated_names_init_vals)

        x.children.append((weight, z, sig))
        return z

    if isinstance(x, AutoDiff):
        names_init_vals = x.get_names_init_vals()
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        val = trace['val']

        updated_trace = {}
        updated_trace.update(trace)

        updated_val = operation(val)

        if np.isnan(updated_val):
                raise ValueError

        updated_trace['val'] = updated_val
        

        # differentiate reverse and forward mode calculations
        if isinstance(x, AutoDiffRev):
            r = AutoDiffRev(names_init_vals=names_init_vals, trace=updated_trace)
            r.grad_val = 1.
            updated_deriv = doperation(val)
            if np.isnan(updated_deriv):
                    raise ValueError
            x.interm_vals.append((updated_deriv, r))
            updated_trace[f'd_{x.name}'] =  x.get_gradient()
            r = AutoDiffRev(names_init_vals=names_init_vals, trace=updated_trace)
        else:
            for var in named_variables:
                updated_deriv = doperation(val) * updated_trace[f'd_{var}']

                if np.isnan(updated_deriv):
                    raise ValueError

                updated_trace[f'd_{var}'] =  updated_deriv 
            r = AutoDiff(names_init_vals=names_init_vals,
                        trace=updated_trace)
        return r

    return operation(x)


def _exp(base):
    def f(x):
        return base**x

    return f

def dexp(base):
    def f(x):
        return base**x * np.log(base)

    return f

def exp(x, base=np.e):
    return __update_unary(x, _exp(base), dexp(base))

def dlog(base):
    def f(x):
        if x <= 0:
            raise ValueError

        return 1./(np.log(base)*x)

    return f

def _log(base):
    def f(x):
        return math.log(x, base)

    return f

def log(x, base=np.e):
    return __update_unary(x, _log(base), dlog(base))

def _logistic(x):
    return 1/(1 + np.exp(-x))

def dlogistic(x):
    return _logistic(x)*(1 - _logistic(x))

def logistic(x):
    return __update_unary(x, _logistic, dlogistic )

def dsqrt(x):
    if x <= 0:
        raise ValueError

    return 1/2 * 1/np.sqrt(x)

def sqrt(x):
    return __update_unary(x, np.sqrt, dsqrt)

def droot(r):
    def f(x):
        return 1./r * x**(1./r - 1)

    return f

def _root(r):
    def f(x):
        return x**(1./r)
    return f

def root(x, r):
    return __update_unary(x, _root(r), droot(r))

def sin(x):
    return __update_unary(x, np.sin, np.cos)

def dcos(x):
    return -np.sin(x)

def cos(x):
    return __update_unary(x, np.cos, dcos)

def dtan(x):
    return 1./(np.cos(x)**2)

def tan(x):
    return __update_unary(x, np.tan, dtan)

def csc(x):
    return 1/sin(x)

def sec(x):
    return 1/cos(x)

def cot(x):
    return 1/tan(x)

def asin(x):
    return arcsin(x)

def darcsin(x):
    return 1/np.sqrt(1 - x**2)

def arcsin(x):
    return __update_unary(x, np.arcsin, darcsin)

def acos(x):
    return arccos(x)

def darccos(x):
    return -1./np.sqrt(1 - x**2)

def arccos(x):
    return __update_unary(x, np.arccos, darccos)
    
def atan(x):
    return arctan(x)

def darctan(x):
    return 1/(1 + x**2)

def arctan(x):
    return __update_unary(x, np.arctan, darctan)

def acsc(x):
    return arccsc(x)

def arccsc(x):
    return asin(1/x)

def asec(x):
    return arcsec(x)

def arcsec(x):
    return acos(1/x)

def acot(x):
    return arccot(x)

def arccot(x):
    return atan(1/x)

def sinh(x):
    return __update_unary(x, np.sinh, np.cosh)

def cosh(x):
    return __update_unary(x, np.cosh, np.sinh)

def dtanh(x):
    return 1/np.cosh(x)**2

def tanh(x):
    return __update_unary(x, np.tanh, dtanh)

def csch(x):
    return 1/sinh(x)

def sech(x):
    return 1/cosh(x)

def coth(x):
    return 1/tanh(x)

def asinh(x):
    return arcsinh(x)

def darcsinh(x):
    return 1/np.sqrt(1 + x**2)

def arcsinh(x):
    return __update_unary(x, np.arcsinh, darcsinh)
    
def acosh(x):
    return arccosh(x)

def darccosh(x):
    return 1/np.sqrt(1 + x) * 1/np.sqrt(x - 1)

def arccosh(x):
    return __update_unary(x, np.arccosh, darccosh)  

def atanh(x):
    return arctanh(x)

def darctanh(x):
    return 1/(1 - x**2)

def arctanh(x):
    return __update_unary(x, np.arctanh, darctanh)  

def acsch(x):
    return arccsch(x)

def arccsch(x):
    return asinh(1/x)

def asech(x):
    return arcsech(x)

def arcsech(x):
    return acosh(1/x)

def acoth(x):
    return arccoth(x)

def arccoth(x):
    return atanh(1/x)






