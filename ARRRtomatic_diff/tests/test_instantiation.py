
#Put in a few different general files (elementary operations, trig, etc.), with no meta-structure below.

#Test regular with standard approach
#Test edge cases with standard approach
#Test TypeErrors with try/catch statements

# from arrrtodiff import AutoDiffVariable
from ARRRtomatic_diff import AutoDiff
# from auto_diff import AutoDiff

# import arrrtodiff.functions as adfuncs

def test_instantiation_neg():
    x = AutoDiff(name='b0', val=-3)
    assert x.trace['val'] == -3,'negative instantiation failed'

def test_instantiation_pos():
    x = AutoDiff(name='b0', val=3)
    assert x.trace['val'] == 3,'positive instantiation failed'

def test_instantiation_zero():
    x = AutoDiff(name='b0', val=0)
    assert x.trace['val'] == 3,'zero instantiation failed'

# def test_string_instantiation():
#     try:
#         x = AutoDiff(name='b0', val="string")
#     except TypeError as e:
#         print("Caught TypeError as expected.")

