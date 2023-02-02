# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from Plotting_routines import *
from RetroRefl import RetroReflector
from DOAS import *
from NIFTy_Magic import *
from create_random_field import *
from Measurement_Devices import *
import matplotlib
from basis_functions_reconstruction import basis_func_reconstruction
from convenience import *

matplotlib.use("TkAgg")

# ---------------------- 3D -----------

# plt.imshow(field, cmap="inferno")
# plt.colorbar()
# plt.show()

#create some 3D DOAS devices:
doas1 = DOAS(1, [0,30,1])
doas2 = DOAS(2, [0,60,1])
doas3 = DOAS(3, [0,0,1])
doas4 = DOAS(4, [30,0,1])
# doas5 = DOAS(5, [0,0,1])
# doas6 = DOAS(6, [0,0,7])
DOASs = [doas1, doas2, doas3, doas4]

#create some 2D DOAS devices:
# doas1 = DOAS(1, [0,150], [1,4])
# doas2 = DOAS(2, [0,290], [1,4])
# doas3 = DOAS(3, [150,0], [10,40])
# doas4 = DOAS(4, [290,0], [10,40])
# doas5 = DOAS(5, [0,0], [10,40])
# DOASs = [doas1, doas2, doas3, doas4, doas5]

#create some 3D Reflectors:
refl1 = RetroReflector(1, [59,59,30])
refl2 = RetroReflector(2, [0,59,30])
refl3 = RetroReflector(3, [20,59,30])
refl4 = RetroReflector(4, [59,20,20])
refl5 = RetroReflector(5, [59,40,5])
refl6 = RetroReflector(6, [59,50,20])
refl7 = RetroReflector(7, [59,59,10])
refl8 = RetroReflector(8, [59,59,20])
refl9 = RetroReflector(9, [20,59,30])
REFLs = [refl1, refl2, refl3, refl4, refl5, refl6, refl7, refl8, refl9]

#create some 2D Reflectors:
# refl1 = RetroReflector(1, [299,299])
# refl2 = RetroReflector(2, [200,290])
# refl3 = RetroReflector(3, [100,290])
# refl4 = RetroReflector(4, [290,200])
# refl5 = RetroReflector(5, [290,100])
# refl6 = RetroReflector(6, [150,290])
# REFLs = [refl1, refl2, refl3, refl4, refl5, refl6]

dimension = "3D"
load_last = True
target_folder = os.path.join("3D_Simulations", "4-doas_9-refl", "gaussexpo", "01-02-2023-baddoas")

main_path, basis_func_path, nifty_path = control_folder_struct(os.getcwd(), target_folder=target_folder, load_last=load_last)

if load_last:
    basis_func_field, xi_basis, mean_rslt, std_rslt, field, final_data_basfunc, final_data_nifty, DOAS_pos, REFL_pos = load_all(main_path, basis_func_path, nifty_path)
    DOASs = [DOAS(i, j) for i, j in enumerate(DOAS_pos)]
    REFLs = [RetroReflector(i, j) for i, j in enumerate(REFL_pos)]
else:
    # field = generate_random_field_nogauss([60,60,30]) #  .val
    field = make_random_3D_field_NIFTy([60,60,30])
    # field = generate_random_field_gauss_expo([60,60,30], 8)

MDevices = Measurement_Devices(DOASs, REFLs, field)

lines = MDevices.return_plottables()

DOAS_positions, REFL_positions = MDevices.return_positions()

measurements, LOS_starts, LOS_ends = MDevices.measure()

lines = MDevices.return_plottables()

if not load_last:
    basis_func_field, xi_basis, final_data_basfunc = basis_func_reconstruction(data=measurements, dim=field.shape, los_starts=LOS_starts, los_ends=LOS_ends, ground_truth=field)
    mean_rslt, std_rslt, master, final_data_nifty = IFT8_reconstruction_3D(field, LOS_starts, LOS_ends, .0001)
    save_all(main_path, basis_func_path, nifty_path, basis_func_field, xi_basis, mean_rslt, std_rslt, field, final_data_basfunc, final_data_nifty, DOAS_positions, REFL_positions)

post_analysis(field, DOASs, REFLs, MDevices, lines, measurements, mean_rslt, std_rslt, basis_func_field, xi_basis, target_folder)

# Plotting(field, DOASs, REFLs, MDevices, lines, measurements, dimension, mean_rslt, std_rslt, master)