import math
import numpy as np

from ARRRtomatic_diff import AutoDiff

x = AutoDiff(name='x', val=np.pi)


print(x.trace['val'])
print(np.cos(x.trace['val']) == -1)


print(x // 2)
print(x % 2)
print(x << 2)
print(x >> 2)
print(x & 2)
print(x ^ 2)
print(x | 2)

print(2 // x)
print(2 % x)
print(2 << x)
print(2 >> x)
print(2 & x)
print(2 ^ x)
print(2 | x)

print(+x)
print(abs(x))
print(~x)
print(complex(x))
print(int(x))
print(float(x))
print(round(x))
print(math.floor(x))
print(math.ceil(x))
print(math.trunc(x))

print(x < 4)
print(x <= 4)
print(x == 4)
print(x != 4)
print(x >= 4)
print(x > 4)

print(4 <  x)
print(4 <= x)
print(4 == x)
print(4 != x)
print(4 >= x)
print(4 > x)

y = AutoDiff(name='y', val=4)


print(repr(x))
print(repr(x+y))

eval(repr(x))

print(x['val'])
x['eee'] = 3

del x['eee']

print(len(x))

print('val' in x)


for a in x:
    print(a)
