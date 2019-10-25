# Organization 
Based off of recommendation by Kenneth Reitz (well known Python developer, author of the requests library)

setup.py
requirements.txt
ARRRtomatic_diff/__init__.py
ARRRtomatic_diff/auto_diff.py
ARRRtomatic_diff/functions/sin.py
ARRRtomatic_diff/functions/...
docs/conf.py
docs/index.rst
tests/test_basic.py
tests/test_advanced.py

.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html

Milestone1 Document

You must clearly outline your software design for the project. This is the main deliverable for this milestone. We are checking to make sure you have a realizable software design, that you understand the scope of the project, and that you understand the details of the project. Here are some sections your group should include in your document along with some prompts that you will want to address.
Introduction

Describe the problem the software solves and why it's important to solve that problem.
Background

Describe (briefly) the mathematical background and concepts as you see fit. You do not need to give a treatise on automatic differentation or dual numbers. Just give the essential ideas (e.g. the chain rule, the graph structure of calculations, elementary functions, etc). Do not copy and paste any of the lecture notes. We will easily be able to tell if you did this as it does not show that you truly understand the problem at hand.
How to Use PackageName

How do you envision that a user will interact with your package? What should they import? How can they instantiate AD objects?

Note: This section should be a mix of pseudo code and text. It should not include any actual operations yet. Remember, you have not yet written any code at this point.
Software Organization

Discuss how you plan on organizing your software package.

    What will the directory structure look like?
    What modules do you plan on including? What is their basic functionality?
    Where will your test suite live? Will you use TravisCI? CodeCov?
    How will you distribute your package (e.g. PyPI)?
    How will you package your software? Will you use a framework? If so, which one and why? If not, why not?
    Other considerations?

Implementation

Discuss how you plan on implementing the forward mode of automatic differentiation.

    What are the core data structures?
    What classes will you implement?
    What method and name attributes will your classes have?
    What external dependencies will you rely on?
    How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?

Be sure to consider a variety of use cases. For example, don't limit your design to scalar functions of scalar values. Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors. Don't forget that people will want to use your library in algorithms like Newton's method (among others).
Document Length

Try to keep your report to a reasonable length. It will form the core of your documentation, so you want it to be a length that someone will actually want to read. Since some of you will use Markdown while others will use Jupyter notebooks and still other group use Latex, we cannot standardize a page length. Use your best judgement. You will only lose points if your document is overly terse (e.g. you do not discuss aspects outlined above) or unbearably long (e.g. you provide so much information that it obscures the message).
Additional Comments

There is no need to have an implementation started for Milestone 1. You are currently in the planning phase of your project. This means that you should feel free to have a project_planning repo in your project organization for scratch work and code.
The actual implementation of your package will start after Milestone 1.
Final Deliverables

There are three primary requirements for this first milestone.

    Create a project organization and invite the teaching staff.
        Within the project organization, create a project repo (make sure teaching staff has access).
        Protect your master branch.
    Create a README.md inside the project repo. At this point, the README should include your the group number, a list of the members on your the team, and badges for Travis CI and CodeCov.
    Fill out this Google form for your group.
    The docs/ directory should include a document called milestone1 (the extension is up to you, but .md or .ipynb are recommended. Details on how to create milestone1 are provided in the Milestone 1 Document section above.
