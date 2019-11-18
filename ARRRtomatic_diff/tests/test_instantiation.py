from ARRRtomatic_diff import AutoDiff


def test_instantiation_neg():
    x = AutoDiff(name='x', val=-3)
    assert x.trace['val'] == -3, 'Negative instantiation failed'
    assert x.trace['d_x'] == 1, 'Negative instantiation failed'


def test_instantiation_pos():
    x = AutoDiff(name='x', val=3.5)
    assert x.trace['val'] == 3.5, 'Positive instantiation failed'
    assert x.trace['d_x'] == 1, 'Positive instantiation failed'


def test_instantiation_zero():
    x = AutoDiff(name='x', val=0)
    assert x.trace['val'] == 0, 'Zero instantiation failed'
    assert x.trace['d_x'] == 1, 'Zero instantiation failed'


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


def test_double_instantiaiton():
    try:
        x = AutoDiff(name='x', val=3, trace=3)
    except ValueError:
        print("Caught error as expected")
