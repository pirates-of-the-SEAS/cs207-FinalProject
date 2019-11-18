"""
This library implements forward mode automatic differentiation for compositions of
elementary operations. It works by defining an "Auto Diff Variable" class, AutoDiff,
that can be used to construct a computational graph corresponding to a
composition of functions that has produces both the value of the composite function
and also all of the partial derivatives with respect to its input variables.

We implement forward mode automatic differentiation through operator overloading
and defining functions corresponding to the elementary mathematical operations
that operate on our AutoDIff objects.

For example, in order to obtain the derivative of x^2 we wouls use the libarary
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
"""

import math

import numpy as np

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
        set of variable names.

    Different AutoDiff objects can be combined through binary operations and
    the gradients are intelligently combined and updated, as long as the
    variables are named appropriately.
    
        INPUTS CONTEXT 1:
        =======
        name: string, the name of the AutoDiff variable
        val: numeric, the value of the AutoDiff variable

        INPUTS CONTEXT 2:
        =======
        name: set, the set of the names of the variables that are the input
              of the AutoDiff object
        trace: dictionary, a dicitonary containing the value and gradient of
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
        >>> x = AutoDiff(trace={'val': 3, 'd_x': 4, 'd_y': 2}, name=set(('x', 'y')))
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
            self.init_variable = True


            self.trace = {
                'val': kwargs['val']
            }

            if 'name' in kwargs:
                self.named_variables = set((kwargs['name'],))
                self.trace['d_{}'.format(kwargs['name'])] = 1
            else:
                raise ValueError("variable name not specified")

        # context 2: construct object assuming trace has been pre-computed
        elif 'trace' in kwargs:
            self.init_variable = False


            self.trace = kwargs['trace']

            if 'name' in kwargs:
                self.named_variables = kwargs['name']
            else:
                raise ValueError("named variables not specified")

    def get_trace(self):
        return self.trace

    def get_named_variables(self):
        return self.named_variables

    def get_value(self):
        return self.trace['val']

    def get_gradient(self):
        return {f'd_{var}':self.trace[f'd_{var}'] for var in self.named_variables}

    @property
    def variables(self):
        return self.get_named_variables()

    @property
    def val(self):
        return self.get_value()

    @property
    def gradient(self):
        return self.get_gradient()

    @staticmethod
    def __verify_same_name_same_value(trace1, trace2):
        pass

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
        other_named_variables = other.get_named_variables()
        named_variables = self.get_named_variables()

        other_trace = other.get_trace()
        trace = self.get_trace()

        combined_named_variables = named_variables | other_named_variables

        val = trace['val'] 
        other_val = other_trace['val']

        # check to see that if we're combining two initial variables
        # of the same name that they have the same value
        if self.init_variable and other.init_variable:
            if (self.named_variables == other.named_variables) and \
               (val != other_val):
                raise Exception("Variables of same name have different values")

        updated_val = update_vals(val, other_val)

        if np.isnan(updated_val):
            raise ValueError

        combined_trace = {'val': updated_val}

        for var in combined_named_variables:
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
                
        return AutoDiff(name=combined_named_variables,
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
        named_variables = self.get_named_variables()
        trace = self.get_trace()
        updated_trace = {}
        updated_trace.update(trace)

        val = updated_trace['val']
         
        updated_val = update_val(val, num)

        if np.isnan(updated_val):
            raise ValueErrr
         
        updated_trace['val'] = updated_val
        for var in named_variables:
            updated_deriv = update_deriv(val,
                                         num,
                                         updated_trace[f'd_{var}'],
                                         0)

            if np.isnan(updated_deriv):
                raise ValueError

            updated_trace[f'd_{var}'] = updated_deriv

        return AutoDiff(name=named_variables,
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
            return self.__update_binary_autodiff(other, AutoDiff.__mul, AutoDiff.__dmul)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__mul, AutoDiff.__dmul)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
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
        return """AutoDiff(name={}, trace={})""".format(
                                repr(self.named_variables),
                                repr(repr(self.trace).strip('"'))
                            )

    def __getitem__(self, key):
        return self.get_trace()[key]

    def __setitem__(self, key, value):
        self.trace[key] = value

    def __delitem__(self, key):
        del self.trace[key]

    def __len__(self):
        return len(self.trace)

    def __contains__(self, item):
        return item in self.trace

    def __iter__(self):
        return iter(self.trace)

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

    def __rlt__(self, other):
        try:
            return other.get_trace()['val'] < self.get_trace()['val']
        except:
            return other < self.get_trace()['val']

    def __rle__(self, other):
        try:
            return other.get_trace()['val'] <= self.get_trace()['val']
        except:
            return other <= self.get_trace()['val']

    def __req__(self, other):
        try:
            return other.get_trace()['val'] == self.get_trace()['val'] 
        except:
            return other == self.get_trace()['val']

    def __rne__(self, other):
        try:
            return other.get_trace()['val'] != self.get_trace()['val'] 
        except:
            return other != self.get_trace()['val'] 

    def __rge__(self, other):
        try:
            return other.get_trace()['val'] >= self.get_trace()['val']
        except:
            return other >= self.get_trace()['val']

    def __rgt__(self, other):
        try:
            return other.get_trace()['val'] > self.get_trace()['val']
        except:
            return other > self.get_trace()['val']

