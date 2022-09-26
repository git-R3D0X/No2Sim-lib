import scipy as scp
import matplotlib.pyplot as plt
from scipy import ndimage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import nifty8 as ift


def generate_random_field_nogauss(dim):
    mean = 0.5
    sigma = 0.3
    gaussian = np.abs(np.random.normal(mean, sigma, size=dim))
    #plt.imshow(gaussian, cmap="inferno")
    #plt.colorbar()
    #plt.show()

    k0 = 0.8
    gamma = 3.5
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
    plt.imshow(np.abs(vol)/np.max(np.abs(vol)), cmap="inferno")
    plt.colorbar()
    plt.show()

    return np.abs(vol)/np.max(np.abs(vol))

def make_random_3D_field_NIFTy(dim, p0=1, offset=3, gamma=3.5):
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
    sky_true = sky(xi_true)

    #plt.imshow(sky_true.val[:,:,15], cmap="inferno")
    #plt.colorbar()
    #plt.show()

    def plotly_plot3D(field):
        """
        Takes a 3D NIFTy field instance and plots it using the plotly library
        :param field:
        :return:
        """
        X, Y, Z = np.mgrid[:field.val.shape[0], :field.val.shape[1], :field.val.shape[2]]
        fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=field.val.flatten(),
            isomin=np.max(field.val)/10,
            isomax=np.max(field.val),
            opacity=0.2,
            surface_count=20,
        ))
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        fig.show()
        #X, Y, Z = np.mgrid[field.val.shape[0], :field.val.shape[1], :field.val.shape[2]]
    #if master:
    #plotly_plot3D(sky_true)
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
