"""
This is a rough implementation of forward mode automatic differentiation for
elementary functions. 

Documentation goes here

https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types
"""
import ARRRtomatic_diff.functions as adfuncs

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

    def __add__(self, other):
        try:
            other_named_variables = other.get_named_variables()
            named_variables = self.get_named_variables()

            other_trace = other.get_trace()
            trace = self.get_trace()

            combined_named_variables = named_variables | other_named_variables

            combined_trace = {
                'val': trace['val'] + other_trace['val']
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

                combined_trace[f'd_{var}'] = d1 + d2

            return AutoDiff(name=combined_named_variables,
                            trace=combined_trace)
        except AttributeError:
            named_variables = self.get_named_variables()
            trace = self.get_trace()
            updated_trace = {}
            updated_trace.update(trace)

            updated_trace['val'] = updated_trace['val'] + other
            
            return AutoDiff(name=named_variables,
                            trace=updated_trace)


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
        return self.__add__(other*-1)

    def __rsub__(self, other):
        """
        other - obj
        """
        return other + self.__mul__(-1) 

    def __mul__(self, other):
         try:
            other_named_variables = other.get_named_variables()
            named_variables = self.get_named_variables()

            other_trace = other.get_trace()
            trace = self.get_trace()

            combined_named_variables = named_variables | other_named_variables

            val1 = trace['val']
            val2 = other_trace['val']

            combined_trace = {
                'val': val1*val2
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

                combined_trace[f'd_{var}'] = d1*val2 + val1*d2

            return AutoDiff(name=combined_named_variables,
                            trace=combined_trace)
         except AttributeError:
            named_variables = self.get_named_variables()
            trace = self.get_trace()
            updated_trace = {}
            updated_trace.update(trace)

            updated_trace['val'] = trace['val']*other
            for var in named_variables:
                updated_trace[f'd_{var}'] = updated_trace[f'd_{var}']*other
            
            return AutoDiff(name=named_variables,
                            trace=updated_trace)



    def __rmul__(self, other):
        return self.__mul__(other)


    def __eq__(self, other):
        raise NotImplementedError

    def __req__(self, other):
        return self.__eq__(other)

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        """
        obj**other
        """
        return adfuncs.exp(other * adfuncs.log(self))
          
        # except AttributeError:
        #     named_variables = self.get_named_variables()
        #     trace = self.get_trace()
        #     updated_trace = {}
        #     updated_trace.update(trace)

        #     updated_trace['val'] = trace['val']**other
        #     for var in named_variables:
        #         updated_trace[f'd_{var}'] = other * (trace['val'])**(other-1) * updated_trace[f'd_{var}']
            
        #     return AutoDiff(name=named_variables,
        #                     trace=updated_trace)

    def __rpow__(self, other):
        """
        other**obj
        """
        return adfuncs.exp(self * adfuncs.log(other))

    def __truediv__(self, other):
        """
        obj/other

        f(x) = g(x)/h(x)

        f'(x) = (g'(x)h(x) - g(x)h'(x)) / h(x)**2
        """
        try:
            other_named_variables = other.get_named_variables()
            named_variables = self.get_named_variables()

            other_trace = other.get_trace()
            trace = self.get_trace()

            combined_named_variables = named_variables | other_named_variables

            val1 = trace['val']
            val2 = other_trace['val']

            combined_trace = {
                'val': val1/val2
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

                combined_trace[f'd_{var}'] = (d1*val2 - val1*d2)/val2**2

            return AutoDiff(name=combined_named_variables,
                            trace=combined_trace)
        except AttributeError:
            named_variables = self.get_named_variables()
            trace = self.get_trace()
            updated_trace = {}
            updated_trace.update(trace)

            updated_trace['val'] = trace['val']/other
            for var in named_variables:
                updated_trace[f'd_{var}'] = updated_trace[f'd_{var}']/other
            
            return AutoDiff(name=named_variables,
                            trace=updated_trace)

    def __rtruediv__(self, other):
        """
        other / obj

        f(x) = g(x)/h(x)

        f'(x) = (g'(x)h(x) - g(x)h'(x)) / h(x)**2
        """
        try:
            other_named_variables = other.get_named_variables()
            named_variables = self.get_named_variables()

            other_trace = other.get_trace()
            trace = self.get_trace()

            combined_named_variables = named_variables | other_named_variables

            val1 = trace['val']
            val2 = other_trace['val']

            combined_trace = {
                'val': val1/val2
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

                combined_trace[f'd_{var}'] = (d2*val1 - val2*d1)/val1**2

            return AutoDiff(name=combined_named_variables,
                            trace=combined_trace)
        except AttributeError:
            named_variables = self.get_named_variables()
            trace = self.get_trace()
            updated_trace = {}
            updated_trace.update(trace)

            updated_trace['val'] = other/trace['val']
            for var in named_variables:
                updated_trace[f'd_{var}'] = -1 * other * updated_trace[f'd_{var}'] / trace['val']**2
            
            return AutoDiff(name=named_variables,
                            trace=updated_trace)

    def __neg__(self):
        return -1 * self


    def __str__(self):
        return str(self.trace)



def main():
    x = AutoDiff(name='x', val=2)
    y = AutoDiff(name='y', val=-5)

    # addition
    print("Addition")
    print(1 + x)
    print(x + 1)

    print(x + x)

    print(x + y)
    print(y + x)

    # multiplication
    print("Multiplication")
    print(3 * x)
    print(x * 3)

    print(x * x)

    print(x * y)
    print(y * x)

    # subtraction
    print("Subtraction")
    print(1 - x)
    print(x - 1)

    print(x - x)

    print(x - y)
    print(y - x)

    # division
    print("Division")
    print(3 / x)
    print(x / 3)

    print(x / x)

    print(x / y)
    print(y / x)

    # exp
    print("Exp")
    print(adfuncs.exp(x))

    # log
    print("log")
    print(x)
    print(adfuncs.log(x))

    # sin
    print("sin")
    print(adfuncs.sin(x))

    # cos
    print("cos")
    print(adfuncs.cos(x))

    # tan
    print("tan")
    print(adfuncs.tan(x))

    # exponentiation
    print("exponentiation")
    print(3 ** x)
    print(x ** 3)
    print((-3) ** x)
    print(x ** (-3))

    print(x ** x)

    # these need to be fixed
    # pow needs to be improved for numerical stability
    print(x ** y)
    print(y ** x)

    z = AutoDiff(name='z', val=8)
    z1 = z**2
    print(z1)
    z2 = adfuncs.sin(z1)
    print(z2)
    z3 = adfuncs.exp(z2)
    print(z3)

    print(adfuncs.exp(x + adfuncs.sin(y) + z))



if __name__ == '__main__':
    main()


