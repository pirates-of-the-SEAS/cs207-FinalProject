[![Build Status](https://travis-ci.org/pirates-of-the-SEAS/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/pirates-of-the-SEAS/cs207-FinalProject.svg?branch=master)

[![codecov](https://codecov.io/gh/pirates-of-the-SEAS/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/pirates-of-the-SEAS/cs207-FinalProject)
  
# Group 16

+ Cameron Hickert
+ Dianne Lee
+ Michael Downs
+ Victor(Wisoo) Song

# About
ARRRtomatic diff is an automatic differentiation library that implements forward mode automatic differentiation. See the docs folder for more detail on how install, setup, and use.
 

# Deadlines

+ Milestone 1: Tuesday, October 29th at 11:59 PM
+ Milestone 2: Tuesday, November 19th
+ Final: Tuesday, December 10th 2019 at 12:00 PM (noon)
+ Showcase: Thursday, December 12th 2019. Location and time TBD 

# CHECKLIST LIST FOR FINAL DELIVERABLE
+ [ ] Your project should be available in your GitHub organization through your project repo.
+ [ ] Your submission should be in the following format:

project_repo/
             README.md
             docs/  
                  documentation
                  milestone1
                  milestone2
             code/
                 ...



+ [ ] Working forward mode implementation
  + [x] Your library should be able to handle real functions of one or more variables. This includes the situation where a user might have multiple functions each of multiple variables.
  + [x] Your library should be able to handle vector functions with multiple real scalar or vector inputs.
+ [ ] Test suite
+ [ ] Updated / extended documentation
+ [ ] New features

+ [ ] The software should be available for download from your GitHub organization.
+ [ ] The software should also be installable from PyPI.
+ [ ] You should provide a requirements.txt file (or something equivalent) with your software so other developers are able to install the necessary dependencies.
+ [ ] After a user installs your package, they should be able to use it without difficulty.


    
+ [ ] Make sure that the following workflow is achievable:
  A user downloads your package from your organization or through PyPI.
  They install the dependencies.
  They run the tests if they're a fellow developer.
  They create a "driver" script in the top level.
  Note: How they interact with your package will depend on your implementation. The interface and other implementation details should be described in your documentation.
  In the driver script, they import your package.
  They instantiate an automatic differentiation object to be used in the forward mode.
  They use the automatic differentiation objects in their own applications (root-finding, optimization, etc).




+ [x] Addition (commutative)
+ [x] Subtraction
+ [x] Multiplication (commutative)
+ [x] Division
+ [x] Power
+ [x] Negation


+ [x]  __lt__ (less than)
+ [x]  __gt__ (greater than)
+ [x]  __le__ (less than or equal to)
+ [x]  __ge__ (greater than or equal to)
+ [x]  __eq__ (equal to)
+ [x]  __ne__ (not equal to)


+ [x]    Trig functions (at the very least, you must have sine, cosine, tangent)
+ [x]    Inverse trig functions (e.g. arcsine, arccosine, arctangent)
+ [x]    Exponentials
        Should be able to handle any base
        You can treat the natural base (e) as a special case
            This is what numpy does.
+ [x]    Hyperbolic functions (sinh, cosh, tanh)
        Note that these can be formed from the natural exponential (e)
+ [x]    Logistic function
        Again, this can be formed from the natural exponential
+ [x]    Logarithms
        Should be able to handle any base.
+ [x]    Square root


+ [ ]  You should have a test suite that runs with pytest. 
+ [ ] Your test suite should run automatically on Travis CI. 
+ [ ] The project GitHub repo should contain a badge showing the pass/fail status of your build. + [ ] The badge should show that your build is passing all tests.

 + [ ] You should also have your project connected to CodeCov. 
 + [ ] Once again, the project repo should have a badge reporting on the coverage of your code from CodeCov. 
 + [ ] Remember: Your code coverage should be at least 90%.


+ [ ] Your documentation must be complete, easy to navigate, and clear.

+ [ ] Remember to update the Background and How to Use sections of your documentation as you add more functionality to your package, so that the user has a good understanding of what he/she can do. 
+ [ ] Call the final form of your documentation documentation.

+ [ ] Your documentation should be a mix of text and hands-on demos.

+ [ ] Introduction 
Describe the problem the software solves and why it is important to solve that problem. This can be built off of the milestones, but you may need to update it depending on what new feature you proposed.
+ [ ] Discussion of why optimization is important
+ [ ] Background 
 + [x] The automatic differentiation background can probably stay the same as in the milestones, unless you were told to update it considerably.
 + [ ] Be sure to include any necessary background for your new feature.
 + [ ] Discussion of optimization routines and reverse mode
 
+ [ ]  How to use your package
    + [ ] How to install?
    + [ ] Include a basic demo for the user. This can be based off of the milestone, but it may change depending on what your new feature is. 
    + [ ] You may want to consider more than one basic demo: one demo just for automatic differentiation and and one demo for your new feature.
    + [ ] Note that this is very much dependent on your final deliverable!
    + [ ] Keep the basic demos to a manageable number.
+ [ ] Software organization
 + [ ] High-level overview of how the software is organized.
 + [ ] Directory structure
 + [ ] Basic modules and what they do
 + [ ] Where do the tests live? How are they run? How are they integrated?
 + [ ] How can someone install your package? Should developers and consumers follow a different installation procedure?
 + [ ] Implementation details
 + [ ] Description of current implementation. This section goes deeper than the high level software organization section.
  + [ ] Try to think about the following:
      + [ ] Core data structures
      + [ ] Core classes
      + [ ] Important attributes
      + [ ] External dependencies
      + [ ] Elementary functions
      
      + [ ] Discussion of AutoDiffVector

+ [ ] Your extension
 + [ ] Description of your extension (the feature(s) you implemented in addition to the minimum requirements.)
 + [ ] Additional information or background needed to understand your extension
 + [ ] This could include required mathematics or other concepts
 
 + [ ] What else do you want to add? What is missing? Don't just think about mathematical things here. Try to think about applications that you'd like to have use your code. Just about every area of science can use automatic differentiation (physics, biology, genetics, applied mathematics, optimization, statistics / machine learning, health science, etc).
 
 + [ ] video
     + [ ] The video should be narrated by all members of your group.
     + [ ] Every group member should speak an equal amount in the video.
     + [ ] For a group of n people, you should change speakers exactly n-1 times.  + [ ] The Introduction / Background and Implementation details/Software organization/How to use sections should contain information related to the minimum project requirements only. 
     + [ ] Introduction/background
     + [ ] Implementation details/Software srganization/How to use
     + [ ] Your additional feature(s) and extension
     + [ ] Future work/possible extensions
     + [ ] Your video should be uploaded to YouTube as a private video. 
     + [ ] Share your video with all of the teaching staff on YouTube. 
     + [ ] You should fill out the video submission Google form which will be sent out by email.
     + [ ] Your video should be at maximum 15 minutes
     + [ ] Make sure the title of your video includes your group number.
     + [ ] Remember, the teaching staff already has full access to your code, so there is no need to focus on small implementation details.
     DO NOT include snippets of your actual library code in your presentation!
     + [ ] Pseudo-code and flowcharts can be very useful to give the big idea of how your package works.
     + [ ] Library demos can be very useful, but be careful. If they don't work well then you'll waste all your video time.
     + [ ] You should provide sufficient background for the project.
     + [ ]
     + Don't overdo the mathematical details for automatic differentation. We are already familiar with them.
     + [ ] Instead, provide the big ideas behind automatic differentation and the motivation for using it.
     + [ ] Spend a fair bit of time on your new feature.
        + [ ] You may need to present some mathematical background to get your audience oriented.
    + [ ] Be sure to conclude with future work and possible extensions.
     


 






+ [ ] python library that can be used for automatic differentiation.
+ [ ] The client should be able to easily install the library, run the tests, access the documentation, and use the library for their application.
+ [ ] Documentation for every subsystem in the project should be provided.
+ [ ] Link to the docs from the README.md in each folder.
+ [ ] The top level README.md should contain an overview, links to other docs, and an installation guide which will help us install and test your system.
+ [ ] For those parts of the project which are modules, python setup.py test should suffice.
+ [ ] For all other parts, include instructions on how to test your code.
+ [ ] Where possible, provide links to Travis-CI test runs and Coveralls coverage.







# Goal
https://harvard-iacs.github.io/2019-CS207/pages/project.html

