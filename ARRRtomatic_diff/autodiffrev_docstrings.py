class AutoDiffRev:
    """AutoDiffRev class implementing reverse mode automatic differentiation through
    operator overloading and explicit construction of the computational graph through
    function composition.
    
    Assumes that the AutoDiffRev object will be initialized in one of two contexts,
    and the argument to the constructor will change depending on the context
    in which the AutoDiff object is created. The user should only ever
    interact with the first context.

    The contexts are: 

    1. A user creating an AutoDiffRev object for the first time in which case the
        arguments 'name' and 'val' are passed. 'name' should only ever be not None
        when initially creating an AutoDiffRev variable. 
       (see example)
    2. A call from inside a function corresponding to an elementary operation,
        in which case the trace is assumed to have been pre-computed and
        the arguments are 'trace' and 'name'. 'name' in this case is a
        dictionary mapping variable name to initial values

    Each AutoDiffRev object epresents a node in the computational graph.
    Additionally, each object maintains a set of breadcrumbs which represent 
    the path to the node. 

    Different AutoDiffRev objects can be combined through binary operations and
    the gradients are intelligently combined and updated, as long as the
    variables are named appropriately.
    
        INPUTS CONTEXT 1:
        =======
        name: string, the name of the AutoDiffRev variable
        val: numeric, the value of the AutoDiffRev variable

        INPUTS CONTEXT 2:
        =======
        names_init_vals: dictionary, a dictionary mapping the names of the variables that
           are the input of the AutoDiffRev object to their initial values. Initial
           values are maintained to enforce consistency.
        trace: dictionary, a dictionary containing the value and gradient of
              the composite function
        
        RETURNS
        ========
        An AutoDiffRev object that maintains the current value of the composite
        function as well as its gradient with respect to the inputs.
    
        EXAMPLES CONTEXT 1:
        =========
        >>> x = AutoDiffRev(name='x', val=2)
        >>> print(x)
        {'val': 2, 'd_x': 1}
        >>>print(5*x)
        {'val': 12, 'd_x': 6}

        EXAMPLES CONTEXT 2:
        =========
        >>> x = AutoDiffRev(trace={'val': 3, 'd_x': 4, 'd_y': 2}, name={'x': 1, 'y': 2})
        >>> print(x)
        {'val': 3, 'd_x': 4, 'd_y': 2}

    """
    def __init__(self, val, name=None, breadcrumbs=None,
                 root_vars=None, names_init_vals=None):
        """
        See the class docstring for an explanation of how this constructor
        method should be called
        """

        # stores name input in two contexts
        self.name = name
        if names_init_vals is None:
            self.names_init_vals = {name: val}
        else:
            self.names_init_vals = names_init_vals

        # toggle that indicates whether the partial derivatives are being
        # computed for this node
        self.__end = False

        # initialize value and intermediate value variables 
        self.val = val
        self.children = []

        # used for memoization later
        self.grad_values = {}
  
        # initialize breadcrumbs and root variables
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
        all of the variables used in the AutoDiffRev object"""
        return self.names_init_vals

    def get_named_variables(self):
        """returns a set containing all of the named variables used in the AutoDiff object"""
        return set(self.names_init_vals.keys())

    def get_value(self):
        """returns the current value of the function"""
        return self.val

    def get_paths(self):
        """returns current path"""
        return set([signature for _, _, signature in self.children])

    @property
    def variables(self):
        return self.get_named_variables()

    @property
    def gradient(self):
        return self.get_gradient()

    @staticmethod
    def __generate_signature():
        """generates random unique hash to define unique var"""
        num_str = str(random.random())
        return hashlib.md5(num_str.encode()).hexdigest().upper()

    @staticmethod
    def __md5(s):
        """returns MD5 hash of input s"""
        return hashlib.md5(s.encode()).hexdigest().upper()

    def __partial(self, breadcrumbs):
        """calculates partial derivative based on breadcrumbs"""
        if self.__end:
            return 1
        
        subset = self.get_paths() & breadcrumbs
        
        return sum(weight * var.__partial(breadcrumbs)
                   for weight, var, signature in self.children
                   if signature in subset)
    
    def get_gradient(self, order=None):
        """Returns a 1D numpy array containing the gradient values as well as the order
        of the variables
            INPUTS
            =======
            order: the order in which the gradient values should be returned

            RETURNS
            ========
            result: a 1D numpy array containing the gradient values
            order: a list containing the variable order. defaults to alphabetical order.

        """
        result = []

        if order is None:
            order = sorted(self.root_vars.keys())
        
        self.__end = True

        for varname in order:
            result.append(self.root_vars[varname].__partial(self.breadcrumbs))
        
        self.__end = False
            
        return result, order
    

    def __update_binary_autodiff(self, other,
                                 update_vals,
                                 first_partial,
                                 second_partial):
        """Combines two AutoDiffRev objects depending on the supplied val and
           derivative update rule. Not to be used externally.
            INPUTS
            =======
            other: AutoDiffRev, the other AutoDiffRev object whose value and
                      gradient will be combined
            update_vals: function, how to combine the values of the two
                                   auto diff object
            first_partial, second_partial: partial derivatives calculated for
                                             self and other 

            RETURNS
            ========
            an AutoDiff object with the combined values and gradients
        """
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


        sig1 = AutoDiffRev.__generate_signature()
        sig2 = AutoDiffRev.__generate_signature()
        
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
        

    def __update_binary_numeric(self, num, update_val, update_deriv):
        pass

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
            return self.__update_binary_numeric(other, AutoDiff.__add, AutoDiff.__dadd)

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
            return self.__update_binary_numeric(other, AutoDiff.__add, AutoDiff.__dadd)


            
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

    # def __repr__(self):
    #     return """AutoDiff(names_init_vals={}, trace={})""".format(
    #                             repr(self.names_init_vals),
    #                             repr(repr(self.trace).strip('"'))
    #                         )

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