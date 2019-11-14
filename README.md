[![Build Status](https://travis-ci.org/pirates-of-the-SEAS/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/pirates-of-the-SEAS/cs207-FinalProject.svg?branch=master)

[![codecov](https://codecov.io/gh/pirates-of-the-SEAS/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/pirates-of-the-SEAS/cs207-FinalProject)

# Group 16

+ Cameron Hickert
+ Dianne Lee
+ Michael Downs
+ Victor(Wisoo) Song


# Milestone 2 todos





+ [ ] Quality Control
    + [ ] Double check the milestone page (https://harvard-iacs.github.io/2019-CS207/project/M2/) to make sure we're not missing any requirements
    + [ ] Go through start to finish process of downloading, installing, setting up, and using package

    + [ ] Adhere to project submission format 

``` 
 project_repo/
             README.md
             docs/  
                  milestone1
                  milestone2
             project_name/
                 ...
```


+ [ ] Working forward mode implementation on scalar values and scalar input functions
    + [ ] Deal with circular imports
    + [ ] Combine functions into a single file
    + [ ] Improve numerical stability of pow operation
    + [ ] Raise exceptions for invalid operations for reals (i.e. division by zero, sqrt of negative, etc.)

    + [ ] Make software available for download from GitHub org

    + [ ] make requirements.txt file (pip freeze, etc.)
    + [ ] Make sure package is easily useable after installation
    + [ ] Make driver script demoing project with Newton's Method example
    
    + [x] Overload addition
    + [x] Overload multiplication
    + [x] Overload subtraction
    + [x] Overload division
    + [x] Overload power
    + [x] Overload negation
    + [ ] Double check for other binary / unary operations
    + [x] implement exponential
    + [x] implement sine
    + [x] implement cosine
    + [x] implement tangent
    

    + [ ] (optional?) implement cosecant
    + [ ] (optional?) implement secant
    + [ ] (optional?) implement cotangent
    + [ ] (optional?) implement tanh
    + [ ] (optional?) implement sinh
    + [ ] (optional?) implement cosh 
    
    + [ ] (optional?) doc strings?
    
    + [ ] (Optional) Remove code redundancy
    + [ ] (Optional) set it up for release on PyPi

    + [ ] (optional) Create wrapper classes for vector valued functions (i.e. broadcasted operations, accessing Jacobian, etc.)


+ [ ] Test suite
    + [ ]  addition
    + [ ]  multiplication
    + [ ]  subtraction
    + [ ]  division
    + [ ]  power
    + [ ]  negation
    + [ ] Negation
    + [ ]  exponential
    + [ ]  sine
    + [ ]  cosine
    + [ ]  tangent
    + [ ] (optional?)  cosecant
    + [ ] (optional?)  secant
    + [ ] (optional?)  cotangent
    + [ ] (optional?)  tanh
    + [ ] (optional?)  sinh
    + [ ] (optional?)  cosh 
    + [ ] edge cases
    + [ ] Make sure all tests pass
    + [ ] Test suite using pytest 
    + [x] integration with travis ci
    + [ ] make sure code is passing all tests
    + [ ] codecov integration
    + [ ] make sure codecov is showing at least 90% code coverage
+ [ ] Updated / extended documentation
    + [ ] Separate milestone 2 documentation into its own file
    + [ ] complete docs folder and documentation
    + [ ] Introduction (copy from milestone 1 documentation)
    + [ ] Emend background (change use of word numerical, add info on Jacobian and seed vector)
    + [ ] Background (copy from milestone 1 documentation)
    + [ ] How to use the package (walk through creation of virtual environment and installation)
    + [ ] How to use the package (basic demo of package)
    + [ ] Updated software organization
        + [ ] Directory structure
        + [ ] What the modules do
        + [ ] Where the tests live
        + [ ] how the rests run
        + [ ] how the tests are integrated
    + [ ] Implementation details
        + [ ] Description of current implementation
            + [ ] Core data structures
            + [ ] Core classes
            + [ ] Important attributes
            + [ ] External dependencies
            + [ ] elementary functions
        + [ ] Discussion of features not yet implemented

    + [ ] (optional) Jupyter notebook with markdown cells for documentation
    + [ ] (optional) use read the docs
+ [ ] Proposal for additional features
 + [ ] write up plans for implementing vector valued functions
     + [ ] How will the software change?
     + [ ] What will the challenges be?
     + [ ] How will the directory structure change?
     + [ ] What new models will be required?
     + [ ] What new classes will be required?
     + [ ] What new data structures will be required?
 + [ ] Decide on advanced feature(s)
     + [ ] write up plans for implementing advanced feature
     + [ ] How will the software change?
     + [ ] What will the challenges be?
     + [ ] How will the directory structure change?
     + [ ] What new models will be required?
     + [ ] What new classes will be required?
     + [ ] What new data structures will be required?

+ [ ] do peer evaluation forms


# Voting for additional features

 How many additional features do we want to implement? One? Two? Three? One per person?
 
 I think that implementing reverse mode is feasible and I have an idea for how to implement it
 based on another class.
 
 I think that implementing back propagation is also feasible.
 
 We could also implement newton's method.
 
 I think that some combination of:
 1. Reverse Mode
 2. Backprop
 3. Newton's Method
 
 would be feasible and give us a decent grade.
 
 
    Implement the reverse mode
    Implement a mixed mode
    Implement back propagation
    Write an application that uses your AD library
        Implicit time-integrator
        Optimization
        Root-finder
    Option for higher-order derivatives (Hessians and beyond)



# Todos

+ [ ] Register group via google form once we have decided on a name: https://docs.google.com/forms/d/e/1FAIpQLSe1pI1Cy0T-ln4niL8O4paK75yFdDiy9B7t8Ze8U3l-t6iyIQ/viewform

+ [ ] Make final presentation video, due on Tuesday December 10th. This presentation will be a demo of your entire library. The final deliverable will be in the form of documentation of your library, including instructions on how to install, run the tests and examples for new users.

# Basic Expectations

+ [ ] python library that can be used for automatic differentiation.
+ [ ] The client should be able to easily install the library, run the tests, access the documentation, and use the library for their application.
+ [ ] Documentation for every subsystem in the project should be provided.
+ [ ] Link to the docs from the README.md in each folder.
+ [ ] The top level README.md should contain an overview, links to other docs, and an installation guide which will help us install and test your system.
+ [ ] For those parts of the project which are modules, python setup.py test should suffice.
+ [ ] For all other parts, include instructions on how to test your code.
+ [ ] Where possible, provide links to Travis-CI test runs and Coveralls coverage.


# Extension ideas
    Implement the reverse mode
    Implement a mixed mode
    Implement back propagation
    Write an application that uses your AD library
        Implicit time-integrator
        Optimization
        Root-finder
    Option for higher-order derivatives (Hessians and beyond)


# Deadlines

+ Milestone 1: Tuesday, October 29th at 11:59 PM
+ Milestone 2: Tuesday, November 19th
+ Final: Tuesday, December 10th 2019 at 12:00 PM (noon)
+ Showcase: Thursday, December 12th 2019. Location and time TBD 



# Goal
https://harvard-iacs.github.io/2019-CS207/pages/project.html
>>>>>>> 1e468afbea8e57a52587d388dd478e855080b949
