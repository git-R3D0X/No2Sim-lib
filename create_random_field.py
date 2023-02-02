import scipy as scp
import matplotlib.pyplot as plt
from scipy import ndimage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import nifty8 as ift

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
            isomin=np.max(field)/30,
            isomax=np.max(field),
            opacity=0.2,
            surface_count=15,
        ))
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        fig.show()
        

def generate_random_field_gauss_expo(dim, cnt_gaussians):
    
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
    
    def create_param_array(cnt_gauss, dim, bounds):
        param_arr = []
        for i in range(cnt_gauss):
            # startvalues at random:
            param_arr.append(np.random.randint(0,high=dim[0],size=1)[0])  # mx
            param_arr.append(np.random.randint(0,high=dim[1],size=1)[0])  # my
            param_arr.append(np.random.uniform(dim[0]/10,dim[0]/2))  # sx
            param_arr.append(np.random.uniform(dim[0]/10,dim[1]/2))  # sy
            param_arr.append(np.random.uniform(0,360))  # theta
        param_arr.append(np.random.uniform(1e-5,0.02))  # random setup for expodecay b
        
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
    
    bounds = [0, dim[0], dim[0]/10, dim[0]/2, 0, 360, 1e-10, 1e-3]
    params, limits = create_param_array(cnt_gaussians, dim, bounds)
    field = make_field(dim, cnt_gaussians, params)
    plotly_plot3D(field)
    return field


def generate_random_field_nogauss(dimension):
    def exponential_decay(b, h): return np.exp(-b*h)
    
    dim = dimension[:2]
    mean = 0.5
    sigma = 0.3
    gaussian = np.abs(np.random.normal(mean, sigma, size=dim))
    #plt.imshow(gaussian, cmap="inferno")
    #plt.colorbar()
    #plt.show()

    k0 = 0.8
    gamma = 5.5
    P0 = 1
    def pspec(P0, k0, gamma, k):
        pspec = P0 * (1+(k/k0)**2)**(-gamma/2)
        return pspec
    pspec_field = np.ones(gaussian.shape)
    mp = [int(gaussian.shape[0]/2), int(gaussian.shape[1]/2)]
    for i in range(gaussian.shape[0]):
        for j in range(gaussian.shape[1]):
            k = [i - mp[0], j - mp[1]]
            pspec_field[i, j] = pspec(P0, k0, gamma, np.sqrt((k[0])**2 + (k[1])**2))

    #plt.imshow(pspec_field/np.max(pspec_field), cmap="inferno")
    #plt.colorbar()
    #plt.show()

    vol = scp.fft.ifft2(np.sqrt(pspec_field)*scp.fft.fft2(gaussian))
    field = np.abs(vol)/np.max(np.abs(vol))

    if np.array(dimension).shape[0] == 3:
        print("HERE")
        h = np.linspace(0, dimension[2], dimension[2])
        b = np.random.uniform(1e-4,0.02)
        field = np.repeat(field[:, :, np.newaxis], dimension[2], axis=2)
        for h in range(1,dimension[2]):
            field[:,:,h] = field[:,:,h-1]*exponential_decay(b,h)
            
    plotly_plot3D(field)

    return field

def make_random_3D_field_NIFTy(dim, p0=1, offset=2, gamma=4.9):
    ift.random.push_sseq_from_seed(np.random.randint(0,100))
    position_space = ift.RGSpace(dim)
    harmonic_space = position_space.get_default_codomain()

    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)

    def sqrtspec(k, p0=p0, offset=offset, gamma=gamma):
        return p0/(offset + k**gamma)

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrtspec)

    A_field = pd(a)
    A = ift.makeOp(A_field)
    xi = ift.FieldAdapter(target=harmonic_space, name='xi') # dict of fields with keys "xi"

    GP = HT @ A @ xi
    xi_true = ift.from_random(GP.domain)

    sky = ift.exp(GP)
    sky_true = sky(xi_true).val

    #if master:
    plotly_plot3D(sky_true)
    return sky_true

def generate_random_field(x_len, y_len, z_len, bubble_amount=15, bubble_scale=5):
    # Generate nicely looking random 3D-field
    np.random.seed(np.random.randint(0,1000))
    X, Y, Z = np.mgrid[:x_len, :y_len, :z_len]
    vol = np.zeros((x_len, y_len, z_len))
    pts = (np.array([x_len * np.random.rand(1, bubble_amount),
                     y_len * np.random.rand(1, bubble_amount),
                     z_len * np.random.rand(1, bubble_amount)])).astype(np.int)
    vol[tuple(indices for indices in pts)] = 1

    vol = ndimage.gaussian_filter(vol, bubble_scale)
    #vol /= vol.max()
    return vol, X, Y, Z

def generate_gauss(x_len, y_len, z_len, bubble_scale=3, centered = False, mux=None, muy=None, muz=None):
    # Generate nicely looking random 3D-field
    X, Y, Z = np.mgrid[:x_len, :y_len, :z_len]
    vol = np.zeros((x_len, y_len, z_len))
    if centered:
        vol[(int(x_len/2), int(y_len/2), int(z_len/2))] = 1
    else:
        vol[mux, muy, muz] = 1
    vol = ndimage.gaussian_filter(vol, bubble_scale)
    vol /= vol.max()
    return vol, X, Y, Z

def generate_random_field2(x_len, y_len, z_len, bubble_amount=6, bubble_scale=5):
    field_tot = np.zeros((x_len, y_len, z_len))
    for i in range(bubble_amount):
        field, X, Y, Z = generate_gauss(x_len, y_len, z_len,
                                        bubble_scale=(np.random.randint(2, 10), np.random.randint(2, 10), np.random.randint(2, 10)),
                                        mux=np.random.randint(0, 29),
                                        muy=np.random.randint(0, 29),
                                        muz=np.random.randint(0, 21))
        field_tot += field

    return field_tot, X, Y, Z
