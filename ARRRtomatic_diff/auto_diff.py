"""
Documentation goes here

https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types
"""

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
            self.trace = trace

            if 'name' in kwargs:
                self.named_variables = kwargs['name']
            else:
                raise ValueError("named variables not specified")

        # case 2: handle initial construction of an auto diff toy object 
        elif 'val' in kwargs:
            self.trace = {
                'val': kwargs['val'],
                f'd_{name}': 1
            }

            if 'name' in kwargs:
                self.named_variables = set((kwargs['name'],))
            else:
                raise ValueError("variable name not specified")


    def get_trace(self):
        return self.trace

    def get_named_variables(self):
        return self.named_variables

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
                    d2 = trace[f'd_{var}']
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
                    d2 = trace[f'd_{var}']
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
        raise NotImplementedError

    def __rpow__(self, other):
        """
        other**obj
        """
        raise NotImplementedError

    def __div__(self, other):
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
                    d2 = trace[f'd_{var}']
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

            updated_trace['val'] = trace['val']/other
            for var in named_variables:
                updated_trace[f'd_{var}'] = updated_trace[f'd_{var}']/other
            
            return AutoDiff(name=named_variables,
                            trace=updated_trace)

    def __rdiv__(self, other):
        raise NotImplementedError

    def __neg__(self):
        return -1 * self


    def __str__(self):
        return "val: {} | deriv: {}".format(self.val, self.der)



def main():
    a = 2.0
    x1 = AutoDiff(a)

    print("Asserting x1 is 2 and deriv is 1")
    assert(x1.val == 2)
    assert(x1.der == 1)
    print("Assertions passed")

    print(x1)

    alpha = 2.0
    beta = 3.0

    print("left Multiplying by 2 and right adding 3:")
    f = alpha * x1 + beta
    assert(f.val == 7.0)
    assert(f.der == 2.)
    print("Assertions passed")
    print(f)


    print("right Multiplying original value by 2 and right adding 3:")
    f = x1 * alpha + beta
    assert(f.val == 7.)
    assert(f.der == 2.)
    print("Assertions passed")
    print(f)

    print("left Multiplying original value by 2 and left adding 3:")
    f = beta + alpha * x1
    assert(f.val == 7.)
    assert(f.der == 2.)
    print("Assertions passed")
    print(f)

    print("right Multiplying original value by 2 and left adding 3:")
    f = beta + x1 * alpha
    assert(f.val == 7.)
    assert(f.der == 2.)
    print("Assertions passed")
    print(f)


    print("Testing operations on  other autodifftoy objects")

    a = 3
    x2 = AutoDiff(a)

    addition_test =  x1 + x2

    assert(addition_test.val == 5)
    assert(addition_test.der == 2)
    print("Assertions passed")
    print(addition_test)

    mult_test = x1 * x2

    assert(mult_test.val == 6)
    assert(mult_test.der == 5)
    print("Assertions passed")
    print(mult_test)

    print("Multiplying product of two autodifftoys by real number and adding real number")

    x3 = 5 * mult_test + 20

    assert(x3.val == 50)
    assert(x3.der == 25)
    print("Assertions passed")
    print(x3)

    print("Left multiplying by autodifftoy and right adding autodifftoy")

    f = x1*x3 + x2

    assert(f.val == 103)
    assert(f.der == 101)
    print("Assertions passed")
    print(f)

    print("Right multiplying by autodifftoy and right adding autodifftoy")

    f = x3*x1 + x2

    assert(f.val == 103)
    assert(f.der == 101)
    print("Assertions passed")
    print(f)

    print("Left multiplying by autodifftoy and left adding autodifftoy")

    f = x2 + x3*x1 

    assert(f.val == 103)
    assert(f.der == 101)
    print("Assertions passed")
    print(f)

    print("Right multiplying by autodifftoy and left adding autodifftoy")

    f = x2 + x1*x3

    assert(f.val == 103)
    assert(f.der == 101)
    print("Assertions passed")
    print(f)

    print("Adding invalid object")
    try:
        "test" + x1
    except Exception as e:
        print("Caught expected exception")
        print(e)

    

if __name__ == '__main__':
    main()


