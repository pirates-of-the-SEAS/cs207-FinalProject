"""
This library implements forward and reverse mode automatic differentiation
for compositions of elementary operations.
It works by defining an "Auto Diff Variable" class, AutoDiff,
that can be used to construct a computational graph corresponding to a
composition of functions that produces both the value of the composite function
and also all of the partial derivatives with respect to its input variables.

There is also an AutoDiffVector class which is a collection of AutoDiff
variables and the analagous AutoDiffRev and AutoDiffVector. 

We implement forward mode automatic differentiation through operator overloading
and defining functions corresponding to the elementary mathematical operations
that operate on our AutoDIff objects.

For example, in order to obtain the derivative of x^2 one would use the libarary
like so:

>>> from ARRRtomatic_diff import AutoDiff
>>> from ARRRtomatic_diff.functions import sin, exp

>>> x = AutoDiff(name='x', val=3)

>>> print(x**2)
{'val': 9, 'd_x': 6.0}

We see that the derivative is maintained along side the value.

Function compositions can be created similarly:

>>> print(sin(x**2))
{'val': 0.4121184852417566, 'd_x': -5.466781571308061}

For vector input functions, we maintain the gradient with respects to the inputs:

>>> y = AutoDiff(name='y', val=2)
>>> z = AutoDiff(name='z', val=-3)

>>> print(sin(x**2)/y + exp(z))
{'val': 0.2558463109887422, 'd_y': -0.10302962131043915, 'd_z': 0.049787068367863944, 'd_x': -2.7333907856540307}

AutoDiffVector objects can be created like sorted
>>> x = AutoDiff(name='x', val=1)
>>> y = AutoDiff(name='y', val=3)
>>> x
AutoDiff(names_init_vals={'x': 1}, trace="{'val': 1, 'd_x': 1}")
>>> y
AutoDiff(names_init_vals={'y': 3}, trace="{'val': 3, 'd_y': 1}")
>>> u = AutoDiffVector([y, -x])
>>> v = AutoDiffVector([x**2, y])
"""

import hashlib

import math

import numpy as np

import random

class AutoDiff:
    """AutoDiff class implementing forward mode automatic differentiation through
    operator overloading and explicit construction of the computational graph through
    function composition.

    Assumes that the AutoDiff object will be initialized in one of two contexts,
    and the argument to the constructor will change depending on the context
    in which the AutoDiff object is created. The user should only ever
    interact with the first context.

    The contexts are: 

    1. A user creating an AutoDiff object for the first time in which case the
        arguments 'name' and 'val' are passed
       (see example)
    2. A call from inside a function corresponding to an elementary operation,
        in which case the trace is assumed to have been pre-computed and
        the arguments are 'trace' and 'name'. 'name' in this case is a
        dictionary mapping variable name to initial values

    Different AutoDiff objects can be combined through binary operations and
    the gradients are intelligently combined and updated, as long as the
    variables are named appropriately.
    
        INPUTS CONTEXT 1:
        =======
        name: string, the name of the AutoDiff variable
        val: numeric, the value of the AutoDiff variable

        INPUTS CONTEXT 2:
        =======
        names_init_vals: dictionary, a dictionary mapping the names of the variables that
           are the input of the AutoDiff object to their initial values. Initial
           values are maintained to enforce consistency.
        trace: dictionary, a dictionary containing the value and gradient of
              the composite function
        
        RETURNS
        ========
        An AutoDiff object that maintains the current value of the composite
        function as well as its gradient with respect to the inputs.
    
        EXAMPLES CONTEXT 1:
        =========
        >>> x = AutoDiff(name='x', val=2)
        >>> print(x)
        {'val': 2, 'd_x': 1}
        >>>print(5*x)
        {'val': 12, 'd_x': 6}

        EXAMPLES CONTEXT 2:
        =========
        >>> x = AutoDiff(trace={'val': 3, 'd_x': 4, 'd_y': 2}, name={'x': 1, 'y': 2})
        >>> print(x)
        {'val': 3, 'd_x': 4, 'd_y': 2}
        """

    def __init__(self, **kwargs):
        """
        See the class docstring for an explanation of how this constructor
        method should be called
        """

        # no parameters specified
        if len(kwargs) == 0:
            raise ValueError("No parameters given")

        # too many parameters specified
        if 'trace' in kwargs and 'val' in kwargs:
            raise ValueError("Both value and trace specified")
 
        # context 1: handle initial construction of an auto diff toy object 
        if 'val' in kwargs:
            self.trace = {
                'val': kwargs['val']
            }

            if 'name' in kwargs:
                self.names_init_vals = {
                    kwargs['name']: kwargs['val']
                }

                self.trace['d_{}'.format(kwargs['name'])] = 1
            else:
                raise ValueError("variable name not specified")

        # context 2: construct object assuming trace has been pre-computed
        elif 'trace' in kwargs:
            self.trace = kwargs['trace']

            if 'names_init_vals' in kwargs:
                self.names_init_vals = kwargs['names_init_vals']
            else:
                raise ValueError("named variables not specified")

    @staticmethod
    def __merge_names_init_vals(d1, d2):
        """Combines two dictionaries mapping variable names to initial values
        and raises an exception if an inconsistency is found. 
            INPUTS
            =======
            d1: the first dictionary
            d2: the second dictionary

            RETURNS
            ========
            a combined dictionary if all initial values are consistent. 

        """
        intersection = d1.keys() & d2

        # verify that the values for keys appearing in both dictionaries are the same
        for name in intersection:
            val1 = d1[name]
            val2 = d2[name]
            if val1 != val2:
                raise Exception("Variable '{}' appears with different values {} and {}".format(
                    name, val1, val2))

        # if they are, return a combined dictionary
        return dict(d1, **d2)

    def copy(self):
        return AutoDiff(names_init_vals=self.names_init_vals,
                        trace=self.trace)

    def get_trace(self):
        """Returns the AutoDiff's trace dictionary"""
        return self.trace

    def get_names_init_vals(self):
        """Returns the dictionary containing the names and initial values of
        all of the variables used in the AutoDiff object"""
        return self.names_init_vals

    def get_named_variables(self):
        """returns a set containing all of the named variables used in the AutoDiff object"""
        return set(self.names_init_vals.keys())

    def get_value(self):
        """returns the current value of the function"""
        return self.trace['val']

    def get_gradient(self, order=None):
        """Returns a 1D numpy array containing the gradient values as well as the order
        of the variables
            INPUTS
            =======
            order: the order in which the gradient values should be returned

            RETURNS
            ========
            g: a 1D numpy array containing the gradinet values
            order: a list containing the variable order. defaults to alphabetical order.

        """

        g = np.zeros(len(self.names_init_vals))

        if order is None:
            order = sorted(self.get_named_variables())

        for i, var in enumerate(order):
            g[i] = self.trace[f'd_{var}']

        return g, order

    @property
    def variables(self):
        return self.get_named_variables()

    @property
    def val(self):
        return self.get_value()

    @property
    def gradient(self):
        return self.get_gradient()

    def __update_binary_autodiff(self, other, update_vals,
                                 update_deriv):
        """Combines two autodiff objects depending on the supplied val and
           derivative update rule. Not to be used externally.
            INPUTS
            =======
            other: AutoDiff, the other AutoDiff object whose value and
                      gradient will be combined
            update_vals: function, how to combine the values of the two
                                   auto diff object
            update_deriv: function, how to combine the gradients of the two
                             auto diff objects

            RETURNS
            ========
            an AutoDiff object with the combined values and gradients
        """

        names_init_vals = self.get_names_init_vals()
        other_names_init_vals = other.get_names_init_vals()

        other_trace = other.get_trace()
        trace = self.get_trace()

        # enforce consistency among the initial values
        combined_names_init_vals = AutoDiff.__merge_names_init_vals(
            names_init_vals, other_names_init_vals)

        val = trace['val'] 
        other_val = other_trace['val']

        # compute the updated value based on the binary operation
        updated_val = update_vals(val, other_val)

        # is typically thrown when an imaginary number appears or there's a
        # division by 0
        if np.isnan(updated_val):
            raise ValueError

        combined_trace = {'val': updated_val}

        # update each partial derivative
        for var in combined_names_init_vals.keys():
            try:
                d1 = trace[f'd_{var}']
            except KeyError:
                d1 = 0
                
            try:
                d2 = other_trace[f'd_{var}']
            except KeyError:
                d2 = 0

            updated_deriv = update_deriv(val, other_val, d1, d2)

            if np.isnan(updated_deriv):
                raise ValueError

            combined_trace[f'd_{var}'] = updated_deriv
                
        return AutoDiff(names_init_vals=combined_names_init_vals,
                        trace=combined_trace)

    def __update_binary_numeric(self, num, update_val, update_deriv):
        """Returns an updated AutoDiff object assuming the current one
                is being used in a binary operation with a numeric.
                Not to be used externally.

            INPUTS
            =======
            num: numeric, a numeric value that will be used to update the
                           value and gradient of the AutoDiff object
            update_vals: function, how to combine the AutoDiff object's val
                                   with the numeric variable
            update_deriv: function, how to combine the AutoDiff object's
                            gradient with the numeric variable

            RETURNS
            ========
            an AutoDiff object with the updated values and gradients
          """
        names_init_vals = self.get_names_init_vals()
        trace = self.get_trace()
        updated_trace = {}
        updated_trace.update(trace)

        val = updated_trace['val']

        # compute the updated value based on the binary operation
        updated_val = update_val(val, num)

        if np.isnan(updated_val):
            raise ValueError

         
        updated_trace['val'] = updated_val
        # update each partial derivative
        for var in names_init_vals:
            updated_deriv = update_deriv(val,
                                         num,
                                         updated_trace[f'd_{var}'],
                                         0)

            if np.isnan(updated_deriv):
                raise ValueError

            updated_trace[f'd_{var}'] = updated_deriv

        return AutoDiff(names_init_vals=names_init_vals,
                            trace=updated_trace)

    @staticmethod
    def __add(x, y):
        return x + y

    @staticmethod
    def __dadd(x, y, dx, dy):
        return dx + dy

    @staticmethod
    def __mul(x, y):
        return x * y

    @staticmethod
    def __dmul(x, y, dx, dy):
        return dx*y + x*dy

    @staticmethod
    def __lpow(x, y):
        return x**y

    @staticmethod
    def __dlpow(x, y, dx, dy):
        if dy == 0:
            return x**(y-1) * (y*dx)
        else:
            return x**(y-1) * (y*dx + x * np.log(x) * dy)

    @staticmethod
    def __rpow(x, y):
        return y**x

    @staticmethod
    def __drpow(x, y, dx, dy):
        if dy == 0:
            return y**x * np.log(y) * dx
        else:
            return y**(x-1)*(x*dy + y * np.log(y) * dx)

    @staticmethod
    def __ldiv(x, y):
        return x / y

    @staticmethod
    def __dldiv(x, y, dx, dy):
        return (dx*y - x*dy)/y**2

    @staticmethod
    def __rdiv(x, y):
        return y / x

    @staticmethod
    def __drdiv(x, y, dx, dy):
        return (dy*x - y*dx)/x**2


    def __add__(self, other):
        try:
            iter(other)
            return AutoDiffVector.combine(self, other, lambda x,y:x+y)
        except:
            pass

        try:
            return self.__update_binary_autodiff(other, AutoDiff.__add, AutoDiff.__dadd)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__add, AutoDiff.__dadd)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        try:
            iter(other)
            return AutoDiffVector.combine(self, other, lambda x,y:x*y)
        except:
            pass


        try:
            return self.__update_binary_autodiff(other, AutoDiff.__mul, AutoDiff.__dmul)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__mul, AutoDiff.__dmul)
            
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        try:
            iter(other)
            return AutoDiffVector.combine(self, other, lambda x,y:x**y)
        except:
            pass

        try:
            return self.__update_binary_autodiff(other, AutoDiff.__lpow, AutoDiff.__dlpow)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__lpow, AutoDiff.__dlpow)
          
    def __rpow__(self, other):
        try:
            return self.__update_binary_autodiff(other, AutoDiff.__rpow, AutoDiff.__drpow)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__rpow, AutoDiff.__drpow)

    def __truediv__(self, other):
        try:
            iter(other)
            return AutoDiffVector.combine(self, other, lambda x,y:x/y)
        except:
            pass

        try:
            return self.__update_binary_autodiff(other, AutoDiff.__ldiv, AutoDiff.__dldiv)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__ldiv, AutoDiff.__dldiv)

    def __rtruediv__(self, other):
        try:
            return self.__update_binary_autodiff(other, AutoDiff.__rdiv, AutoDiff.__drdiv)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__rdiv, AutoDiff.__drdiv)

    def __neg__(self):
        return -1 * self

    def __bool__(self):
        return bool(self.get_trace()['val'])

    def __repr__(self):
        return """AutoDiff(names_init_vals={}, trace={})""".format(
                                repr(self.names_init_vals),
                                repr(repr(self.trace).strip('"'))
                            )

    def __str__(self):
        return str(self.trace)

    def __floordiv__(self, other):
        return self.get_trace()['val'] // other

    def __mod__(self, other):
        return self.get_trace()['val'] % other

    def __lshift__(self, other):
        return self.get_trace()['val'] << other

    def __rshift__(self, other):
        return self.get_trace()['val'] >> other

    def __and__(self, other):
        return self.get_trace()['val'] & other

    def __xor__(self, other):
        return self.get_trace()['val'] ^ other

    def __or__(self, other):
        return self.get_trace()['val'] | other

    def __rfloordiv__(self, other):
        return other // self.get_trace()['val']

    def __rmod__(self, other):
        return other % self.get_trace()['val'] 

    def __rlshift__(self, other):
        return other << self.get_trace()['val']

    def __rrshift__(self, other):
        return other >> self.get_trace()['val']

    def __rand__(self, other):
        return other & self.get_trace()['val'] 

    def __rxor__(self, other):
        return other ^ self.get_trace()['val']

    def __ror__(self, other):
        return other | self.get_trace()['val']

    def __pos__(self):
        return self

    def __abs__(self):
        return abs(self.get_trace()['val'])

    def __round__(self):
        return round(self.get_trace()['val'])

    def __floor__(self):
        return math.floor(self.get_trace()['val'])

    def __ceil__(self):
        return math.ceil(self.get_trace()['val'])

    def __trunc__(self):
        return math.trunc(self.get_trace()['val'])

    def __invert__(self):
        return ~self.get_trace()['val']

    def __complex__(self):
        return complex(self.get_trace()['val'])

    def __int__(self):
        return int(self.get_trace()['val'])

    def __float__(self):
        return float(self.get_trace()['val'])

    def __lt__(self, other):
        try:
            return self.get_trace()['val'] < other.get_trace()['val']
        except:
            return self.get_trace()['val'] < other

    def __le__(self, other):
        try:
            return self.get_trace()['val'] <= other.get_trace()['val']
        except:
            return self.get_trace()['val'] <= other

    def __eq__(self, other):
        try:
            return self.get_trace()['val'] == other.get_trace()['val']
        except:
            return self.get_trace()['val'] == other

    def __ne__(self, other):
        try:
            return self.get_trace()['val'] != other.get_trace()['val']
        except:
            return self.get_trace()['val'] != other

    def __ge__(self, other):
        try:
            return self.get_trace()['val'] >= other.get_trace()['val']
        except:
            return self.get_trace()['val'] >= other

    def __gt__(self, other):
        try:
            return self.get_trace()['val'] > other.get_trace()['val']
        except:
            return self.get_trace()['val'] > other


class AutoDiffVector:
    """AutoDiffVector class that takes as input an iterable of AutoDiff objects
  or numeric primitives and keeps track of all named variables. Is largely a
  convenience interface for defining vector valued functions, performing
  bradcasting operations on vector valued functions, and computing the
  Jacobian. In the current implementation, it is not intended for the user
  to be able to change the number of elements in the vector. 

    """
    def __init__(self, auto_diff_variables):
        """Constructor for AutoDiffVector. 

            INPUTS
            =======
            num: auto_diff_variables, an iterable of AutoDiff objects and
                  numeric primitives
            
        """

        # maintain variable names
        try:
            self.num_funcs = len(auto_diff_variables)

            if self.num_funcs == 0:
                raise Exception("AutoDiffVector cannot be empty")


            self.__auto_diff_variables = list(auto_diff_variables)


            # get all of the variable names for each AutoDiff object
            self.named_variables = set({})
            for ad in auto_diff_variables:
                try:
                    
                    self.named_variables |= ad.get_named_variables()
                except:
                    continue

        except:
            raise Exception("AutoDiffVector requires an iterable as input")

        self.idx = 0


    def __len__(self):
        return self.num_funcs

    def __iter__(self):
        return self.copy()

    def __next__(self):
        """iterable with respect to auto diff variables"""
        if self.idx < self.num_funcs:
            result = self.__auto_diff_variables[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration

    def get_named_variables(self):
        """return set of all named variables in all AutoDiff objects making up the
        AutoDiffVector"""
        return self.named_variables

    def get_values(self):
        """
        returns a 1D numpy array containing all values of all AutoDiff objects
        """
        results = []

        for ad in self.__auto_diff_variables:
            try:
                results.append(ad.trace['val'])
            except  AttributeError:
                results.append(ad)

        return np.array(results)

    def get_jacobian(self, order=None):
        """
        returns a 2D numpy array where the ith row is the gradient of the ith
        AutoDiff object where the variables appear in a specified order.
        """
        num_vars = len(self.named_variables)

        # in the case where there are no named variables e.g
        # the vector consists of numerics, return a zero column vector 

        if num_vars == 0:
            num_vars = 1
            order = ['x']
        elif order is None:
            order = sorted(self.named_variables)

        J = np.zeros((self.num_funcs, num_vars))
        
        for i, ad in enumerate(self.__auto_diff_variables):
            try:
                g_i, _ = ad.get_gradient(order)
                J[i, :] = g_i
            except AttributeError:
                pass

        return J, order

    def copy(self):
        return AutoDiffVector(self.__auto_diff_variables)

    @property
    def variables(self):
        return self.get_named_variables()

    @property
    def val(self):
        return self.get_values()

    @property
    def jacobian(self):
        return self.get_jacobian()

    @staticmethod
    def combine(first, other, operation):
        """Combines two objects that are some combination of AutoDiff,
        AutoDiffVector, or numeric primitive. Performs scalar or vector
        operations where appropriate. 

            INPUTS
            =======
            first: either a numeric primitive, AutoDiff, or AutoDiffVector
            other: either a numeric primitive, AutoDiff, or AutoDiffVector
            operation: a binary operation on two numeric primitives

            RETURNS
            ========
            An AutoDiffVector containing the result of the scalar or vector operation
            
        """
        result = []

        # both are iterables of the same length
        try:
            if len(first) != len(other):
                raise Exception("Dimentionality mismatch: {} vs {}".format(
                    len(first), len(other)))

            for var1, var2 in zip(first, other):

                result.append(operation(var1, var2))

            return AutoDiffVector(result)

        except TypeError:
            pass

        # first is an iterable and other is a scalar
        try:
            for var in first:
                result.append(operation(var, other))

            return AutoDiffVector(result)
        except TypeError:
            pass

        # second is an iterable and first is a scalar

        try:
            for var in other:
                result.append(operation(first, var))

            return AutoDiffVector(result)
        except TypeError:
            pass

        # both are scalars
        try:
            result.append(operation(first, other))

            return AutoDiffVector(result)
        except TypeError:
            pass

        raise ValueError

    def apply_to_vals(self, operation):
        """Broadcasts a unary operation to each element in the
           AutoDiffVector

            INPUTS
            =======
            operation: unary operation

            RETURNS
            ========
            An AutoDiffVector containing the result of the broadcasted unary opeartion
        """
        result = []

        try:
            for v in self.__auto_diff_variables:
                result.append(operation(v))

            return result

        except:
            raise ValueError


    def dot(self, other):
        """computes a dot product with another iterable of the same dimension

            INPUTS
            =======
            other: another iterable

            RETURNS
            ========
            the result of the dot product. A numeric primiitive or AutoDifF
        """
        if len(self) != len(other):
                raise Exception("Dimentionality mismatch: {} vs {}".format(
                    len(self), len(other)))

        result = 0
        for var1, var2 in zip(self, other):
            result += var1*var2

        return result

    def sq_norm(self):
        return self.dot(self)

    def __add__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x+y)

    def __radd__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x+y)

    def __sub__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x-y)

    def __rsub__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x-y)

    def __mul__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x*y)

    def __rmul__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x*y)

    def __pow__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x**y)
          
    def __rpow__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x**y)

    def __truediv__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x/y)

    def __rtruediv__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x/y)

    def __neg__(self):
        return -1 * self

    def __bool__(self):
        return self.apply_to_vals(bool)

    def __repr__(self):
        repr_str = "["
        for var in self.__auto_diff_variables:
            repr_str += repr(var)
            repr_str += ','

        repr_str = repr_str[:-1]

        repr_str += ']'

        return """AutoDiffVector(names_init_vals={}""".format(
                                repr_str
                            )

    def __getitem__(self, idx):
        return self.__auto_diff_variables[idx].copy()

    # current design choice is for AutoDiffVector to not be mutable
    def __setitem__(self, key, value):
        raise NotImplementedError

    # current design choice is for AutoDiffVector to not be mutable
    def __delitem__(self, key):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError

    def __str__(self):
        vec_str = "["

        for var in self.__auto_diff_variables:
            try:
                vec_str += str(var.get_trace())
            except AttributeError:
                vec_str += str(var)
            vec_str += ','

        vec_str = vec_str[:-1]

        vec_str += ']'
        return vec_str

    def __floordiv__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x // y).get_values()

    def __mod__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x % y).get_values()

    def __lshift__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x << y).get_values()

    def __rshift__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x << y).get_values()

    def __and__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x & y).get_values()

    def __xor__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x ^ y).get_values()

    def __or__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x | y).get_values()

    def __rfloordiv__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x // y).get_values()

    def __rmod__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x % y).get_values()

    def __rlshift__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x << y).get_values()

    def __rrshift__(self, other):
        return AutoDiffVector.combine(self, other, lambda x,y: x << y).get_values()

    def __rand__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x & y).get_values()

    def __rxor__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x ^ y).get_values()

    def __ror__(self, other):
        return AutoDiffVector.combine(other, self, lambda x,y: x | y).get_values()

    def __pos__(self):
        return self

    def __abs__(self):
        return self.apply_to_vals(abs)

    def __round__(self):
        return self.apply_to_vals(round)

    def __floor__(self):
        return self.apply_to_vals(math.floor)

    def __ceil__(self):
        return self.apply_to_vals(math.ceil)

    def __trunc__(self):
        return self.apply_to_vals(math.trunc)

    def __invert__(self):
        return self.apply_to_vals(lambda x: ~x)

    def __complex__(self):
        return self.apply_to_vals(complex)

    def __int__(self):
        return self.apply_to_vals(int)

    def __float__(self):
        return self.apply_to_vals(float)

    def __lt__(self, other):
        return all(AutoDiffVector.combine(self, other, lambda x,y: x < y).get_values())

    def __le__(self, other):
        return all(AutoDiffVector.combine(self, other, lambda x,y: x <= y).get_values())

    def __eq__(self, other):
        return all(AutoDiffVector.combine(self, other, lambda x,y: x == y).get_values())

    def __ne__(self, other):
        return all(AutoDiffVector.combine(self, other, lambda x,y: x != y).get_values())

    def __ge__(self, other):
        return all(AutoDiffVector.combine(self, other, lambda x,y: x >= y).get_values())

    def __gt__(self, other):
        return all(AutoDiffVector.combine(self, other, lambda x,y: x > y).get_values())


class AutoDiffRev:
    """
    represents a node in the computational graph.
    maintains a set of breadcrumbs which represent the path to the node

    name should only ever be not none when initially creating
    an AutoDiffRev variable

    """
    def __init__(self, val, name=None, breadcrumbs=None,
                 root_vars=None, names_init_vals=None):

        # name
        self.name = name
        if names_init_vals is None:
            self.names_init_vals = {name: val}
        else:
            self.names_init_vals = names_init_vals

        # toggle that indicates whether the partial derivatives are being
        # computed for this node
        self.__end = False

        # 
        self.val = val

        # 
        self.children = []

        # will possibly be used in memoization later
        self.grad_values = {}
  
        if breadcrumbs is None:
            self.breadcrumbs = set({})
        else:
            self.breadcrumbs = breadcrumbs
            
        if root_vars is None:
            self.root_vars = {name: self}
        else:
            self.root_vars = root_vars 



    @staticmethod
    def __merge_names_init_vals(d1, d2):
        """Combines two dictionaries mapping variable names to initial values
        and raises an exception if an inconsistency is found. 
            INPUTS
            =======
            d1: the first dictionary
            d2: the second dictionary

            RETURNS
            ========
            a combined dictionary if all initial values are consistent. 

        """
        intersection = d1.keys() & d2

        # verify that the values for keys appearing in both dictionaries are the same
        for name in intersection:
            val1 = d1[name]
            val2 = d2[name]
            if val1 != val2:
                raise Exception("Variable '{}' appears with different values {} and {}".format(
                    name, val1, val2))

        # if they are, return a combined dictionary
        return dict(d1, **d2)

    def get_names_init_vals(self):
        """Returns the dictionary containing the names and initial values of
        all of the variables used in the AutoDiff object"""
        return self.names_init_vals

    def get_named_variables(self):
        """returns a set containing all of the named variables used in the AutoDiff object"""
        return set(self.names_init_vals.keys())

    def get_value(self):
        """returns the current value of the function"""
        return self.val



    @property
    def variables(self):
        return self.get_named_variables()


    def get_paths(self):
        return set([signature for _, _, signature in self.children])


    @property
    def gradient(self):
        return self.get_gradient()

    @staticmethod
    def generate_signature():
        num_str = str(random.random())
        return hashlib.md5(num_str.encode()).hexdigest().upper()

    @staticmethod
    def __md5(s):
        return hashlib.md5(s.encode()).hexdigest().upper()

    def __partial(self, breadcrumbs):
        if self.__end:
            return 1
        
        subset = self.get_paths() & breadcrumbs
        
                
        return sum(weight * var.__partial(breadcrumbs)
                   for weight, var, signature in self.children
                   if signature in subset)
    
    def get_gradient(self, order=None):
        result = []

        if order is None:
            order = sorted(self.root_vars.keys())
        
        self.__end = True

        for varname in order:
            result.append(self.root_vars[varname].__partial(self.breadcrumbs))
        
        self.__end = False
            
        return np.array(result), order
    

    def __update_binary_autodiff(self, other,
                                 update_vals,
                                 first_partial,
                                 second_partial):

        names_init_vals = self.get_names_init_vals()
        other_names_init_vals = other.get_names_init_vals()

        # verify that all variables have the same initial value
        updated_names_init_vals = AutoDiffRev.__merge_names_init_vals(
            names_init_vals, other_names_init_vals)

        val = self.get_value()
        other_val = other.get_value()

        updated_val = update_vals(val, other_val)

        # is typically thrown when an imaginary number appears or there's a
        # division by 0
        if np.isnan(updated_val):
            raise ValueError


        sig1 = AutoDiffRev.generate_signature()
        sig2 = AutoDiffRev.generate_signature()
        
        updated_breadcrumbs = self.breadcrumbs | \
                              other.breadcrumbs| \
                              set([sig1])             | \
                              set([sig2])     
        
        # keep track of root variables
        updated_root_vars = {}
        updated_root_vars.update(self.root_vars)
        updated_root_vars.update(other.root_vars)

        # keep track of paths
        z = AutoDiffRev(val=updated_val,
                        breadcrumbs=updated_breadcrumbs,
                        root_vars=updated_root_vars,
                        names_init_vals=updated_names_init_vals)


        self.children.append((first_partial, z, sig1))
        other.children.append((second_partial, z, sig2))

        return z
        

    def __update_binary_numeric(self, num, update_vals, partial):
        updated_names_init_vals = self.get_names_init_vals()

        val = self.get_value()

        updated_val = update_vals(val, num)

        # is typically thrown when an imaginary number appears or there's a
        # division by 0
        if np.isnan(updated_val):
            raise ValueError

        sig = AutoDiffRev.generate_signature()
        
        updated_breadcrumbs = self.breadcrumbs | set([sig])     
        
        # keep track of root variables
        updated_root_vars = self.root_vars.copy()

        # keep track of paths
        z = AutoDiffRev(val=updated_val,
                        breadcrumbs=updated_breadcrumbs,
                        root_vars=updated_root_vars,
                        names_init_vals=updated_names_init_vals)


        self.children.append((partial, z, sig))

        return z

    @staticmethod
    def __add(x, y):
        return x + y

    @staticmethod
    def __dadd(x, y, dx, dy):
        return dx + dy

    @staticmethod
    def __mul(x, y):
        return x * y

    @staticmethod
    def __dmul(x, y, dx, dy):
        return dx*y + x*dy

    @staticmethod
    def __lpow(x, y):
        return x**y 

    @staticmethod
    def __dlpow(x, y, dx, dy):
        if dy == 0:
            return x**(y-1) * (y*dx)
        else:
            return x**(y-1) * (y*dx + x * np.log(x) * dy)

    @staticmethod
    def __rpow(x, y):
        return y**x

    @staticmethod
    def __drpow(x, y, dx, dy):
        if dy == 0:
            return y**x * np.log(y) * dx
        else:
            return y**(x-1)*(x*dy + y * np.log(y) * dx)

    @staticmethod
    def __ldiv(x, y):
        return x / y

    @staticmethod
    def __dldiv(x, y, dx, dy):
        return (dx*y - x*dy)/y**2

    @staticmethod
    def __rdiv(x, y):
        return y / x

    @staticmethod
    def __drdiv(x, y, dx, dy):
        return (dy*x - y*dx)/x**2

    def __add__(self, other):
        # try:
        #     iter(other)
        #     return AutoDiffVector.combine(self, other, lambda x,y:x+y)
        # except:
        #     pas
        try:
            return self.__update_binary_autodiff(other,
                                                 AutoDiffRev.__add,
                                                 1,
                                                 1)
        except AttributeError:
            return self.__update_binary_numeric(other,
                                         AutoDiffRev.__add,
                                         1)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        # try:
        #     iter(other)
        #     return AutoDiffVector.combine(self, other, lambda x,y:x*y)
        # except:
        #     pass

        try:
            return self.__update_binary_autodiff(other,
                                                 AutoDiffRev.__mul,
                                                 other.get_value(),
                                                 self.get_value())
        except AttributeError:
            return self.__update_binary_numeric(other,
                                         AutoDiffRev.__mul,
                                                other)


            
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        # try:
        #     iter(other)
        #     return AutoDiffVector.combine(self, other, lambda x,y:x**y)
        # except:
        #     pass

        
        selfval = self.get_value()
        
        try:
            otherval = other.get_value()
            return self.__update_binary_autodiff(other,
                                                 AutoDiffRev.__lpow,
                                                 otherval*selfval**(otherval-1),
                                                 selfval**otherval * np.log(selfval)
                                                 )
        except AttributeError:
            return self.__update_binary_numeric(other,
                                         AutoDiffRev.__lpow,
                                                other * selfval**(other-1))
          
    def __rpow__(self, other):
        selfval = self.get_value()
        
        return self.__update_binary_numeric(other,
                                         AutoDiffRev.__rpow,
                                                selfval**other * np.log(selfval))
                                                

    def __truediv__(self, other):
        selfval = self.get_value()
        
        try:
            otherval = other.get_value()
            return self.__update_binary_autodiff(other,
                                                 AutoDiffRev.__ldiv,
                                                 1/otherval,
                                                 -1 * selfval / otherval**2
                                                 )
        except AttributeError:
            return self.__update_binary_numeric(other,
                                         AutoDiffRev.__ldiv,
                                                1/other)

    def __rtruediv__(self, other):
        selfval = self.get_value()
        
        return self.__update_binary_numeric(other,
                                         AutoDiffRev.__rdiv,
                                                -1 * other / selfval**2)

    def __neg__(self):
        return self * -1

    def __bool__(self):
        return bool(self.get_value())

    # def __repr__(self):
    #     return """AutoDiff(names_init_vals={}, trace={})""".format(
    #                             repr(self.names_init_vals),
    #                             repr(repr(self.trace).strip('"'))
    #                         )


    def __floordiv__(self, other):
        return self.get_value() // other

    def __mod__(self, other):
        return self.get_value() % other

    def __lshift__(self, other):
        return self.get_value() << other

    def __rshift__(self, other):
        return self.get_value() >> other

    def __and__(self, other):
        return self.get_value() & other

    def __xor__(self, other):
        return self.get_value() ^ other

    def __or__(self, other):
        return self.get_value() | other

    def __rfloordiv__(self, other):
        return other // self.get_value()

    def __rmod__(self, other):
        return other % self.get_value() 

    def __rlshift__(self, other):
        return other << self.get_value()

    def __rrshift__(self, other):
        return other >> self.get_value()

    def __rand__(self, other):
        return other & self.get_value() 

    def __rxor__(self, other):
        return other ^ self.get_value()

    def __ror__(self, other):
        return other | self.get_value()

    def __pos__(self):
        return self

    def __abs__(self):
        return abs(self.get_value())

    def __round__(self):
        return round(self.get_value())

    def __floor__(self):
        return math.floor(self.get_value())

    def __ceil__(self):
        return math.ceil(self.get_value())

    def __trunc__(self):
        return math.trunc(self.get_value())

    def __invert__(self):
        return ~self.get_value()

    def __complex__(self):
        return complex(self.get_value())

    def __int__(self):
        return int(self.get_value())

    def __float__(self):
        return float(self.get_value())

    def __lt__(self, other):
        try:
            return self.get_value() < other.get_value()
        except:
            return self.get_value() < other

    def __le__(self, other):
        try:
            return self.get_value() <= other.get_value()
        except:
            return self.get_value() <= other

    def __eq__(self, other):
        try:
            return self.get_value() == other.get_value()
        except:
            return self.get_value() == other

    def __ne__(self, other):
        try:
            return self.get_value() != other.get_value()
        except:
            return self.get_value() != other

    def __ge__(self, other):
        try:
            return self.get_value() >= other.get_value()
        except:
            return self.get_value() >= other

    def __gt__(self, other):
        try:
            return self.get_value() > other.get_value()
        except:
            return self.get_value() > other

             
    def diagnose(self):
        print("value")
        print(self.val)
        
        print("root vars")
        print(self.root_vars)
        
        print("breadcrumbs")
        print(self.breadcrumbs)
        
        print("children")
        for weight, var, signature in self.children:
            print(signature)
        

    
    def __str__(self):
        return f"{self.val}"


