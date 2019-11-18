from ARRRtomatic_diff import AutoDiff

def test_instantiation_neg():
    x = AutoDiff(name='b0', val=-3)
    assert x.trace['val'] == -3,'negative instantiation failed'

def test_instantiation_pos():
    x = AutoDiff(name='b0', val=3)
    assert x.trace['val'] == 3,'positive instantiation failed'

def test_instantiation_zero():
    x = AutoDiff(name='b0', val=0)
    assert x.trace['val'] == 0,'zero instantiation failed'

def test_bogus_instantiation():
    try:
        x = AutoDiff("gobbledgook")
    except TypeError:
        print("Caught error as expected")

def test_empty_instantiation():
    try:
        x = AutoDiff()
    except ValueError:
        print("Caught error as expected")

# def test_overfull_instantiation():
#     try:
#         x = AutoDiff(name="b0", val=3, val=5)
#     except SyntaxError:
#         print("Caught error as expected")