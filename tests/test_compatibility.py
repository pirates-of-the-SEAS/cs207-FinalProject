from ARRRtomatic_diff import *


def test_Diff_RevDiff_add():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x+y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_mul():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x*y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_sub():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x - y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_pow():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x ** y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_true_div():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x / y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_true_div():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x // y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_mod():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x % y
    except TypeError as e:
        print(e)

def test_Diff_RevDiff_lshift():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    try:
        x << y
    except TypeError as e:
        print(e)


def test_Diff_RevDiff_gt():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)

    assert x > y

def test_Diff_RevDiff_lt():
    x = AutoDiff(name='x', val=2)
    y = AutoDiffRev(name='y', val=5)

    assert x < y


