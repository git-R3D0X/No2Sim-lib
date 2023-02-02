import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math as m
import os
from scipy.optimize import minimize
import plotly.graph_objects as go
import nifty8 as ift
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
# ---- TEX SETUP ----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5

def plotly_plot3D(field):
        """
        Takes a 3D NIFTy field instance and plots it using the plotly library
        :param field:
        :return:
        """
        X, Y, Z = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
        fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=field.flatten(),
            isomin=np.max(field)/10,
            isomax=np.max(field),
            opacity=0.2,
            surface_count=10,
        ))
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        fig.show()

def gaussian(mx, my, sx, sy, A, x, y):
    return A / (2. * np.sqrt(np.pi * sx * sy)) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def gaussian_rotatable(mux, muy, sigmax, sigmay, theta, grid_x, grid_y):
        theta = np.deg2rad(theta)
        val = np.exp(-0.5 * ((((grid_x - mux) * np.cos(theta) -
                               (grid_y - muy) * np.sin(theta)) / sigmax) ** 2 + (((grid_x - mux) * np.sin(theta) +
                                                                                  (grid_y - muy) * np.cos(theta)) / sigmay) ** 2)) # * 1/(sigmax * 2 * np.pi * sigmay)
        return val

def exponential_decay(b, h):
    return np.exp(-b*h)

def make_field(dim, cnt_gaussians, params):
    baseplate = np.zeros([dim[0], dim[1]])
    for i in range(cnt_gaussians):
        mux, muy, sigamx, sigmay, theta = params[i:i+5]
        x = np.linspace(0, dim[0], dim[0])
        y = np.linspace(0, dim[1], dim[1])
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        z = gaussian_rotatable(mux, muy, sigamx, sigmay, theta, x, y)
        baseplate += z
    
    h = np.linspace(0, dim[2], dim[2])
    b = params[-1]
    field = np.repeat(baseplate[:, :, np.newaxis], dim[2], axis=2)
    for h in range(1,dim[2]):
        field[:,:,h] = field[:,:,h-1]*exponential_decay(b,h)
    return field

def data_diff(params, *args):
    # gaussians on the [::0] axis, let decay act on every vertical column with field[x,y,0] = expodec(0)
    # gaussian base-plate:
    try:
        dim, real_data, cnt_gaussians, los_starts, los_ends, debug = args
    except ValueError:
        dim, real_data, cnt_gaussians, los_starts, los_ends, debug = args[0]
    
    field = make_field(dim, cnt_gaussians, params)
    
    position_space = ift.RGSpace(field.shape)
    ift_field = ift.makeField(domain=position_space, arr=field)
    R = ift.LOSResponse(position_space, starts=los_starts, ends=los_ends)
    tmp_data = R(ift_field).val
    
    if debug:
        print(tmp_data)
        plotly_plot3D(field)
    tmp_data_save = tmp_data
    Xi = np.sum(np.abs(tmp_data - real_data)**2)
    # print(f"Xi²: {Xi}")
    Xi_basis_func.append(Xi)
    return Xi

def create_param_array(cnt_gauss, dim, bounds):
    param_arr = []
    for i in range(cnt_gauss):
        # startvalues at random:
        param_arr.append(np.random.randint(0,high=dim[0],size=1)[0])  # mx
        param_arr.append(np.random.randint(0,high=dim[1],size=1)[0])  # my
        param_arr.append(np.random.uniform(dim[0]/10,dim[0]/2))  # sx
        param_arr.append(np.random.uniform(dim[0]/10,dim[1]/2))  # sy
        param_arr.append(np.random.uniform(0,360))  # theta
    param_arr.append(np.random.uniform(0,0.1))  # random setup for expodecay b
    
    # set limits:
    mu_min, mu_max, sigma_min, sigma_max, theta_min, theta_max, b_min, b_max = bounds
    limits_arr = []
    for i in range(cnt_gauss):
        limits_arr.append((mu_min, mu_max))
        limits_arr.append((mu_min, mu_max))
        limits_arr.append((sigma_min, sigma_max))
        limits_arr.append((sigma_min, sigma_max))
        limits_arr.append((theta_min, theta_max))
    limits_arr.append((b_min, b_max))
    return np.array(param_arr), np.array(limits_arr)
    
def basis_func_reconstruction(data, dim, los_starts, los_ends, ground_truth, showres=False):
    global Xi_basis_func
    global tmp_data_save
    tmp_data_save = 0
    Xi_basis_func = []
    debug=False
    cnt_gaussians = m.floor(data.shape[0]/5)-2  # gauss takes 5 params, expo takes 2
    bounds = [0, dim[0], dim[0]/10, dim[0]/2, 0, 360, 1e-4, 0.02]
    params, limits = create_param_array(cnt_gaussians, dim, bounds)

    # for exponent in np.linspace(0,0.5,5):
    #     params[-1] = exponent
    #     plotly_plot3D(make_field(dim, cnt_gaussians, params))
    #     #plt.plot(exponential_decay(params[-1], exponent, np.linspace(0,40,40)), label=f"expo: {round(exponent, 3)}")
    # #plt.legend()
    # #plt.show()
    
    data_diff(params, dim, data, cnt_gaussians, los_starts, los_ends, debug)
    res = minimize(data_diff, params, args=[dim, data, cnt_gaussians, los_starts, los_ends, debug], bounds=limits)
    print(res)
    
    
    if showres:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.xaxis.set_tick_params(which='major', size=9, width=1.5, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=4, width=1.5, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=9, width=1.5, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=4, width=1.5, direction='in', right='on')
        plt.ylim([0,np.max(np.array(Xi_basis_func)) + np.max(np.array(Xi_basis_func))/10])
        plt.title(r"$\chi^2\mathrm{\,\,Gauss-Expo\,\,Reconstruction}$")
        plt.ylabel(r"$\chi^2$")
        plt.xlabel(r"$\mathrm{Step}$")
        plt.grid()
        plt.plot(Xi_basis_func)  # np.linspace(0,len(Xi_basis_func), len(Xi_basis_func)), 
        plt.show()
        
        plotly_plot3D(ground_truth)
        plotly_plot3D(make_field(dim, cnt_gaussians, params=res.x))
    
    return make_field(dim, cnt_gaussians, params=res.x), Xi_basis_func, tmp_data_save


































