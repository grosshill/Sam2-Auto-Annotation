from maths import Quaternion as q
import torch as th
from math import sqrt

pos_global = th.tensor([0, 1, 0], dtype =th.float32)

pos_local = th.tensor([1, 0, 0], dtype =th.float32)

rot = q(sqrt(2) / 2, .0, .0 , sqrt(2) / 2)
print(rot)

print(rot.rotate(pos_local))
print(rot.inv_rotate(pos_local))