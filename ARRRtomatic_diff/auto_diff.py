"""
This is a rough implementation of forward mode automatic differentiation for
elementary functions. 

Documentation goes here

https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types
"""

import math

import numpy as np

class AutoDiff:
    def __init__(self, **kwargs):
        """
        name:
        val:
        trace:
        """

        # no parameters specified
        if len(kwargs) == 0:
            raise ValueError("No parameters given")

        # too many parameters specified

        if 'trace' in kwargs and 'val' in kwargs:
            raise ValueError("Both value and trace specified")
 
        # case 1: construct object assuming trace has been pre-computed
        if 'trace' in kwargs:
            self.trace = kwargs['trace']

            if 'name' in kwargs:
                self.named_variables = kwargs['name']
            else:
                raise ValueError("named variables not specified")

        # case 2: handle initial construction of an auto diff toy object 
        elif 'val' in kwargs:
            self.trace = {
                'val': kwargs['val']
            }

            if 'name' in kwargs:
                self.named_variables = set((kwargs['name'],))
                self.trace['d_{}'.format(kwargs['name'])] = 1
            else:
                raise ValueError("variable name not specified")


    def get_trace(self):
        return self.trace

    def get_named_variables(self):
        return self.named_variables

    def get_value(self):
        return self.trace['val']

    def get_gradient(self):
        return {f'd_{var}':self.trace[f'd_{var}'] for var in self.named_variables}

    def __update_binary_autodiff(self, other, update_vals,
                                 update_deriv):
        other_named_variables = other.get_named_variables()
        named_variables = self.get_named_variables()

        other_trace = other.get_trace()
        trace = self.get_trace()

        combined_named_variables = named_variables | other_named_variables

        val = trace['val'] 
        other_val = other_trace['val']

        updated_val = update_vals(val, other_val)

        if np.isnan(updated_val):
            raise ValueError

        combined_trace = {
            'val': updated_val
        }

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
        return x**(y-1) * (y*dx + x * np.log(x) * dy)

    @staticmethod
    def __rpow(x, y):
        return y**x

    @staticmethod
    def __drpow(x, y, dx, dy):
        return y*(x-1)*(y*dx + np.log(y) + x*dy)

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
        """__radd__ is only called if the left object does not have an __add__
        method, or that method does not know how to add the two objects (which
        it flags by returning NotImplemented). Both classes have an __add__
        method, which do not return NotImplemented. Therefore the __radd__
        method would never be called.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        obj - other
        """
        return self + -other

    def __rsub__(self, other):
        """
        other - obj
        """
        return other + -self

    def __mul__(self, other):
        try:
            return self.__update_binary_autodiff(other, AutoDiff.__mul, AutoDiff.__dmul)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__mul, AutoDiff.__dmul)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        """
        obj**other
        """
        try:
            return self.__update_binary_autodiff(other, AutoDiff.__lpow, AutoDiff.__dlpow)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__lpow, AutoDiff.__dlpow)
          
    def __rpow__(self, other):
        """
        other**obj
        """
        try:
            return self.__update_binary_autodiff(other, AutoDiff.__rpow, AutoDiff.__drpow)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__rpow, AutoDiff.__drpow)

    def __truediv__(self, other):
        """
        obj/other

        f(x) = g(x)/h(x)

        f'(x) = (g'(x)h(x) - g(x)h'(x)) / h(x)**2
        """
        try:
            return self.__update_binary_autodiff(other, AutoDiff.__ldiv, AutoDiff.__dldiv)
        except AttributeError:
            return self.__update_binary_numeric(other, AutoDiff.__ldiv, AutoDiff.__dldiv)

    def __rtruediv__(self, other):
        """
        other / obj

        f(x) = g(x)/h(x)

        f'(x) = (g'(x)h(x) - g(x)h'(x)) / h(x)**2
        """
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

