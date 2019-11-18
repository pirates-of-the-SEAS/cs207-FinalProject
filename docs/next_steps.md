# Vector valued functions
We plan to extend our current implementation to be able to handle vector valued functions. We will do so by creating a convenience class, AutoDiffVector, that takes a collection of AutoDiff objects and supports broadcasting operations (i.e. scalar operations or vector arithmetic when performing algebraic operations with other AutoDiffVector objects). This class will perform the bookkeeping that keeps track of all of the variables across each function and will also have convenience methods for accessing the Jacobian. We expect that this addition will entail adding a new class to auto_diff.py and the software structure will otherwise not change. Possible challeneges include figuring out design considerations such as whether we will want the AutoDiffVector's dimensions to be mutable.

Example usage:

```python
from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff.functions import sin, exp, sqrt

x = AutoDiff(name='x', val=-1)
y = AutoDiff(name='y', val=3)

u = AutoDiffVector((x,y))
v = AutoDiffVector((-y,x))

# performs vector addition, scalar multiplication, and broadcasts the 
# unary operator sin element-wise
z = sin(5*(u + v))

# a numpy array representnug the jacobina of the vector-valued function
# f1 = x - y
@ f2 = y + x
J = z.get_jacobian() 
```


# Newton's method
We plan to implement Newton's method and possibly other root finding / optimization routines such as SGD (and the special case of backpropagation), time permitting. for an arbitrary vector valued function. We will do so by creating a subpackage, optimization, which will contain modules that implement the optimization routines via functions. For Newton's Method, we will compute the Jacobian/derivative through our AutoDiffVector/AutoDiff objects and use numpy to solve the linear system in the multivariate setting. We do not foresee many challeneges in implementing newton's method. 

Example usage:
```python
from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin
from ARRRtomatic_diff.optimization import newton

def f(x):
    x = AutoDiff(name='x', val=x)

    auto_diff_results = sin(x)

    return auto_diff_results['val'], auto_diff_results['d_x']
    
do_newtons_method(0.2, f, verbose=1)
Iteration 1 | x: 0.200000 | f(x): 0.198669 | deriv: 0.980067
Iteration 2 | x: -0.002710 | f(x): -0.002710 | deriv: 0.999996
Converged to 6.634450606078646e-09 after 3 iterations
```

# Reverse mode
We plan to implement the reverse mode as an alternative to forward mode. We will do so by creating another AutoDiff class, AutoDiffRev, and the corresponding reverse AutoDiffVector, AutoDiffVectorRev. The API will be mostly the same - the computational graph will be explicitly constructed via algebraically combining AutoDiffRev variables. The intermediate forward pass values and partials will be calculated through these operations as well as the graph dependency structure. For example:

```python
from ARRRtomatic_diff import AutoDiffRev

x = Var(1)
y = Var(2)
z = x * y
z.partial = 1.0

print(z.value)
print(x.partial)
```

In the above code, the multiplication of x and y keeps track of the forward pass evaluations and immediate partials and also specifies z as a child node of x and y in the dependency graph. Each AutoDiffRev variable will keep track of the operations that it's used in. Unlike our forward mode implementation, dz/dx and dz/dy are maintained in the x and y variables, so they must be kept around. The computation of the partial derivatives of the output with respect to the input is performed recursively. 

The primary challenges to implementing reverse mode will be making the API elegant for the vector-valued case, working with a different mental model for automatic differentiation, and making our reverse mode implementation work with our optimization routines. Our implementation of reverse mode will require adding two additioanl classes, AutoDiffRev and AutoDiffVectorRev to auto_diff.py. We do not expect to add any new modules 


# Checks to ensure that variables of the same name have the same initial value
Currently, we have a rudimentary check to verify that user isn't attempting to combine two AutoDiff variables that have the same named variable but with differing values. It's still possible for a user to create an AutoDiff object with a named variable, perform operations on it, and then combing it with another AutoDiff object with the same named variable but with a different value. We plan to make our software throw an exception when this happens. This will require modifying AutoDiff to keep track of the initial values for each variable and then perform a check when attempting to combine two AutoDiff objects.

Example:

```python
from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin

x1 = AutoDiff(name='x', val=1)
x2 = AutoDiff(name='x', val=3)

# will throw an exception
x1 + x2

@ does not throw an exception but should
(x1 + x1) + x2
```



