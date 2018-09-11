import numpy as np
from IPython import embed
import math
def calc_relative_bearing(me, them, my_bearing):
    dy = them[0]-me[0]
    dx = them[1]-me[1]
    bearing = np.rad2deg(math.atan2(dy,dx))
    my_diff_from_90 = (90-my_bearing)%360
    out_bearing = (bearing+my_diff_from_90)%360
    return out_bearing

#         90
#  135 \  | / 45
#   180  ---  0
#  225  / | \ 315
#        270

print("up")
print(calc_relative_bearing(me=(5,3), them=(10,3), my_bearing=90))
print("should go left")
print(calc_relative_bearing(me=(5,3), them=(10,3), my_bearing=45))
print("should go right")
print(calc_relative_bearing(me=(5,3), them=(10,3), my_bearing=135))
print("should go right")
print(calc_relative_bearing(me=(5,3), them=(10,3), my_bearing=180))

print("should go right")
print(calc_relative_bearing(me=(5,3), them=(5,10), my_bearing=90))
print("should go straight")
print(calc_relative_bearing(me=(5,3), them=(5,10), my_bearing=0))
print("should go left")
print(calc_relative_bearing(me=(5,3), them=(5,10), my_bearing=270))


print("should go down")
print(calc_relative_bearing(me=(5,3), them=(1,3), my_bearing=90))
print("should go right")
print(calc_relative_bearing(me=(5,3), them=(1,3), my_bearing=0))
print("should go left")
print(calc_relative_bearing(me=(5,3), them=(1,3), my_bearing=180))
print("should go straight")
print(calc_relative_bearing(me=(5,3), them=(1,3), my_bearing=270))




embed()


