import os
import numpy as np
import matplotlib.pyplot as plt
# ---- TEX SETUP ----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5

def control_folder_struct(cwd, target_folder, load_last):
    if os.path.exists(os.path.join(cwd, target_folder)):
        if not load_last:
            print(f"OS ERROR - PATH ALREADY EXISTS: {os.path.join(cwd, target_folder)}\n")
            inp = input("CONTINUE ANYWAY? [Y/N]")
            if inp=="Y" or inp=="y":
                main_path = os.path.join(cwd, target_folder)
                basis_func_path = os.path.join(cwd, target_folder, "basis_func")
                nifty_path = os.path.join(cwd, target_folder, "nifty")
            else:
                exit()
        else:
            main_path = os.path.join(cwd, target_folder)
            basis_func_path = os.path.join(cwd, target_folder, "basis_func")
            nifty_path = os.path.join(cwd, target_folder, "nifty")
    else:
        main_path = os.mkdir(os.path.join(cwd, target_folder))
        basis_func_path = os.mkdir(os.path.join(cwd, target_folder, "basis_func"))
        nifty_path = os.mkdir(os.path.join(cwd, target_folder, "nifty"))
    return str(main_path), str(basis_func_path), str(nifty_path)

def save_all(main_path, basis_func_path, nifty_path, basis_func_field, xi_basis, mean_rslt, std_rslt, field, final_data_basfunc, final_data_nifty, DOAS_positions, REFL_positions):
    try:
        np.save(os.path.join(basis_func_path, "basis_field.npy"), basis_func_field)
        np.save(os.path.join(basis_func_path, "basis_xi.npy"), np.array(xi_basis))
        np.save(os.path.join(basis_func_path, "basis_data_res.npy"), np.array(final_data_basfunc))
        np.save(os.path.join(nifty_path, "nifty_mean.npy"), mean_rslt)
        np.save(os.path.join(nifty_path, "nifty_std.npy"), std_rslt)
        np.save(os.path.join(nifty_path, "nifty_data_res.npy"), final_data_nifty)
        np.save(os.path.join(main_path, "ground_truth_field.npy"), field)
        np.save(os.path.join(main_path, "DOASs.npy"), np.array(DOAS_positions))
        np.save(os.path.join(main_path, "REFLs.npy"), np.array(REFL_positions))
    except FileNotFoundError:
        main_path, basis_func_path, nifty_path = control_folder_struct(os.getcwd(), target_folder="simulation_results", load_last=False)
        np.save(os.path.join(basis_func_path, "basis_field.npy"), basis_func_field)
        np.save(os.path.join(basis_func_path, "basis_xi.npy"), np.array(xi_basis))
        np.save(os.path.join(basis_func_path, "basis_data_res.npy"), np.array(final_data_basfunc))
        np.save(os.path.join(nifty_path, "nifty_mean.npy"), mean_rslt)
        np.save(os.path.join(nifty_path, "nifty_std.npy"), std_rslt)
        np.save(os.path.join(nifty_path, "nifty_data_res.npy"), final_data_nifty)
        np.save(os.path.join(main_path, "ground_truth_field.npy"), field)
        np.save(os.path.join(main_path, "DOASs.npy"), np.array(DOAS_positions))
        np.save(os.path.join(main_path, "REFLs.npy"), np.array(REFL_positions))
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.xaxis.set_tick_params(which='major', size=9, width=1.5, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=4, width=1.5, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=9, width=1.5, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=4, width=1.5, direction='in', right='on')
    try:
        plt.ylim([0,np.max(np.array(xi_basis)) + np.max(np.array(xi_basis))/10])
    except ValueError:
        plt.ylim([0,200])
    plt.title(r"$\chi^2\mathrm{\,\,Gauss-Expo\,\,Reconstruction}$")
    plt.ylabel(r"$\chi^2$")
    plt.xlabel(r"$\mathrm{Step}$")
    plt.grid()
    plt.plot(xi_basis)  # np.linspace(0,len(Xi_basis_func), len(Xi_basis_func)), 
    plt.savefig(str(os.path.join(basis_func_path, "xi_res.png")))

def load_all(main_path, basis_func_path, nifty_path):
    basis_func_field = np.load(os.path.join(basis_func_path, "basis_field.npy"))
    xi_basis = np.load(os.path.join(basis_func_path, "basis_xi.npy"))
    final_data_basfunc = np.load(os.path.join(basis_func_path, "basis_data_res.npy"))
    mean_rslt = np.load(os.path.join(nifty_path, "nifty_mean.npy"))
    std_rslt = np.load(os.path.join(nifty_path, "nifty_std.npy"))
    final_data_nifty = np.load(os.path.join(nifty_path, "nifty_data_res.npy"))
    field = np.load(os.path.join(main_path, "ground_truth_field.npy"))
    DOAS_positions = np.load(os.path.join(main_path, "DOASs.npy"), allow_pickle=True)
    REFL_positions = np.load(os.path.join(main_path, "REFLs.npy"), allow_pickle=True)
    return basis_func_field, xi_basis, mean_rslt, std_rslt, field, final_data_basfunc, final_data_nifty, DOAS_positions, REFL_positions

