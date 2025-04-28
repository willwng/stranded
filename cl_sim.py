'''
Input: set of centerlines on a scalp

Algorithm:
- normalize relative positions of root point from each CL on the scalp on some surface
- compute negative density gradient for each CL root point on the surface
- calculate force on each CL point based on density gradient
   - force magnitude falls off with distance from root point, but has same directionality
   - root points DO NOT MOVE, but other points do
- update CL points based on force
- repeat until equilibrium configuration is reached (loss will be some "energy" related to the centerline force)

Output: set of CLs with updated positions
'''

import numpy as np
from rod.rod_generator import RodGenerator

# generating a set of CLs
poses = []
thetas = []
for i in range(5):
    pos, theta = RodGenerator.straight_rod(50)
    for i in range(pos.shape[0]):
        pos[i, 0] += 5 * i # spacing out x coords
    poses.append(pos)
    thetas.append(theta)

roots = np.array([pos[0] for pos in poses])

# generating a surface


