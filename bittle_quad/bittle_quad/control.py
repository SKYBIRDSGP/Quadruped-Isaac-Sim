import numpy as np

left_posvec = np.array([30, 30, 30, 30])
right_posvec = np.array([-30, -30, -30, -30])

np.save("left_posvec.npy",left_posvec)
posvec = np.load("left_posvec.npy")

np.save("right_posvec.npy",right_posvec)
posvec = np.load("right_posvec.npy")

print(left_posvec, right_posvec)