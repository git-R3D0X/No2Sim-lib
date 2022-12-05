# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from Plotting_routines import *
from RetroRefl import RetroReflector
from DOAS import *
from NIFTy_Magic import *
from create_random_field import *
from Measurement_Devices import *
import matplotlib

matplotlib.use("TkAgg")

# ---------------------- 3D -----------

dimension = "2D"
field = generate_random_field_nogauss([300, 300]) #  .val
plt.imshow(field, cmap="inferno")
plt.colorbar()
plt.show()

#create some 3D DOAS devices:
#doas1 = DOAS(1, [0,20,1], [1,4,20])
#doas2 = DOAS(2, [0,39,4], [1,4,200])
#doas3 = DOAS(3, [20,0,2], [10,40,200])
#doas4 = DOAS(4, [39,0,4], [10,40,200])
#doas5 = DOAS(5, [0,0,1], [10,40,200])
#doas6 = DOAS(6, [0,0,7], [10,40,200])
#DOASs = [doas1, doas2, doas3, doas4, doas5, doas6]

#create some 2D DOAS devices:
doas1 = DOAS(1, [0,150], [1,4])
doas2 = DOAS(2, [0,290], [1,4])
doas3 = DOAS(3, [150,0], [10,40])
doas4 = DOAS(4, [290,0], [10,40])
doas5 = DOAS(5, [0,0], [10,40])
DOASs = [doas1, doas2, doas3, doas4, doas5]

#create some 3D Reflectors:
#refl1 = RetroReflector(1, [39,39,7])
#refl2 = RetroReflector(2, [0,39,7])
#refl3 = RetroReflector(3, [10,39,7])
#refl4 = RetroReflector(4, [39,10,5])
#refl5 = RetroReflector(5, [39,20,2])
#refl6 = RetroReflector(6, [39,30,5])
#refl7 = RetroReflector(7, [39,39,2])
#refl8 = RetroReflector(8, [39,39,5])
#refl9 = RetroReflector(9, [10,39,7])
#REFLs = [refl1, refl2, refl3, refl4, refl5, refl6, refl7, refl8, refl9]

#create some 2D Reflectors:
refl1 = RetroReflector(1, [299,299])
refl2 = RetroReflector(2, [200,290])
refl3 = RetroReflector(3, [100,290])
refl4 = RetroReflector(4, [290,200])
refl5 = RetroReflector(5, [290,100])
refl6 = RetroReflector(6, [150,290])
REFLs = [refl1, refl2, refl3, refl4, refl5, refl6]


MDevices = Measurement_Devices(DOASs, REFLs, field)

lines = MDevices.return_plottables()

DOAS_positions, REFL_positions = MDevices.return_positions()

measurements, LOS_starts, LOS_ends = MDevices.measure()

lines = MDevices.return_plottables()

mean_rslt, std_rslt, master = IFT8_reconstruction_2D(field, LOS_starts, LOS_ends, .0001)

Plotting(field, DOASs, REFLs, MDevices, lines, measurements, dimension, mean_rslt, std_rslt, master)