# -*- coding: utf-8 -*-

import numpy as np
import nifty8 as ift
import plotly.graph_objects as go
import os

def IFT8_reconstruction_2D(field, LOS_starts, LOS_ends, noise_lvl):
    #doas_pos = return_positions(dim="2D")[0]
    #refl_pos = return_positions(dim="2D")[1]
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        master = comm.Get_rank() == 0
    except ImportError:
        comm = None
        master = True

    if os.path.exists("./output_ift_inversion"):
        pass
    else:
        os.mkdir("./output_ift_inversion")

    filename = "./output_ift_inversion/testing_field_inversion_{}.png"
    position_space = ift.RGSpace((field.shape[0], field.shape[1]))
    normalized_field = field
    padder = ift.FieldZeroPadder(position_space, [i * 2 for i in position_space.shape], central=False).adjoint

    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),
        # Amplitude of field fluctuations
        'fluctuations': (1., 0.1),  # 1.0, 1e-2
        # Exponent of power law power spectrum component
        'loglogavgslope': (-4., 1),  # -6.0, 1   (mean, std)
        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (0.5, 0.5),  # 1.0, 0.5
        # How ragged the integrated Wiener process component is
        'asperity': (0.01, 0.01)  # 0.1, 0.5
    }
    #
    correlated_field = ift.SimpleCorrelatedField(padder.domain, **args)
    pspec = correlated_field.power_spectrum

    # Apply a nonlinearity
    signal = ift.log(ift.Adder(1., domain=position_space)(ift.exp(padder(correlated_field))))

    # Build the line-of-sight response and define signal response
    print(LOS_starts)
    print("\n")
    print(LOS_ends)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = noise_lvl
    N = ift.ScalingOperator(data_space, noise, np.float64)

    # Create the field instance for the gt field
    ground_truth = ift.makeField(position_space, normalized_field)
    data = R(ground_truth) + N.draw_sample()

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
                                               deltaE=0.005, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.05,
                                             convergence_level=2, iteration_limit=5)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                                  deltaE=0.05, iteration_limit=15, convergence_level=2)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=N.inverse) @ signal_response)
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    plot = ift.Plot()
    plot.add(ground_truth, title='Ground Truth', zmin=0, zmax=1)
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec.force(mean)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

    # number of samples used to estimate the KL
    # Minimize KL
    n_iterations = 6
    n_samples = lambda iiter: 5 if iiter < 5 else 10
    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                              minimizer, ic_sampling, None,
                              plottable_operators={"signal": (signal, dict(vmin=0, vmax=1)),
                                                   "power spectrum": pspec},
                              ground_truth_position=None,
                              output_directory="output_ift_inversion",
                              overwrite=True, comm=comm)

    if True:
        # Load result from disk. May be useful for long inference runs, where
        # inference and posterior analysis are split into two steps
        samples = ift.ResidualSampleList.load("output_ift_inversion/pickle/last", comm=comm)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    mean, var = samples.sample_stat(signal)
    plot.add(mean, title="Posterior Mean", vmin=0, vmax=1)
    plot.add(var.sqrt(), title="Posterior Standard Deviation", vmin=0)

    nsamples = samples.n_samples
    logspec = pspec.log()
    #plot.add(list(samples.iterator(pspec)) +
    #         [pspec.force(ground_truth), samples.average(logspec).exp()],
    #         title="Sampled Posterior Power Spectrum",
    #         linewidth=[1.] * nsamples + [3., 3.],
    #         label=[None] * nsamples + ['Ground truth', 'Posterior mean'])
    return mean.val.T, var.sqrt().val.T, master


def IFT8_reconstruction_3D(field, LOS_starts, LOS_ends, noise_lvl):

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
            isomin=0.5,
            isomax=np.max(field),
            opacity=0.2,
            surface_count=20,
        ))
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        fig.show()


    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        master = comm.Get_rank() == 0
    except ImportError:
        comm = None
        master = True

    if os.path.exists("./output_ift_inversion"):
        pass
    else:
        os.mkdir("./output_ift_inversion")

    filename = "./output_ift_inversion/testing_field_inversion_{}.png"

    position_space = ift.RGSpace((field.shape))
    padder = ift.FieldZeroPadder(position_space, [i * 2 for i in position_space.shape], central=False).adjoint

    args = {
        'offset_mean': 3,
        'offset_std': (3, 2),

        # Amplitude of field fluctuations ()
        'fluctuations': (2., 0.5),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-6, 0.5),  # -6.0, 1   (mean, std)

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (0.5, 0.3),  # 1.0, 0.5 (wiggle around the linear line )

        # How ragged the integrated Wiener process component is
        'asperity': (0.1, 0.1)  # 0.1, 0.5
    }

    correlated_field = ift.SimpleCorrelatedField(padder.domain, **args)
    pspec = correlated_field.power_spectrum

    # ift.random.push_sseq_from_seed(np.random.randint(0,100))

    # Apply a nonlinearity
    signal = ift.log(ift.Adder(1., domain=position_space)(ift.exp(padder(correlated_field))))

    # Build the line-of-sight response and define signal response
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    noise = noise_lvl
    N = ift.ScalingOperator(data_space, noise, np.float64)

    # Generate mock signal and data
    ground_truth = ift.makeField(position_space, field)

    data = R(ground_truth) + N.draw_sample()

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
                                               deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
                                             convergence_level=2, iteration_limit=5)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                                  deltaE=0.5, iteration_limit=15, convergence_level=2)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = ift.GaussianEnergy(data, inverse_covariance=N.inverse) @ signal_response
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)


    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    plot = ift.Plot()
    plot.add(ground_truth, title='Ground Truth', zmin=0, zmax=1)
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec.force(mean)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

    # number of samples used to estimate the KL
    # Minimize KL
    n_iterations = 8
    n_samples = lambda iiter: 3 if iiter < 5 else 8
    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                              minimizer, ic_sampling, minimizer_sampling,
                              plottable_operators={"signal": (signal, dict(vmin=0, vmax=1)),
                                                   "power spectrum": pspec},
                              ground_truth_position=None,
                              output_directory="output_ift_inversion",
                              overwrite=True, comm=comm)

    if True:
        # Load result from disk. May be useful for long inference runs, where
        # inference and posterior analysis are split into two steps
        samples = ift.ResidualSampleList.load("output_ift_inversion/pickle/last", comm=comm)
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
                isomin=0.5,
                isomax=np.max(field),
                opacity=0.2,
                surface_count=20,
            ))
            fig.update_layout(scene_xaxis_showticklabels=False,
                              scene_yaxis_showticklabels=False,
                              scene_zaxis_showticklabels=False)
            fig.show()
        #if master:
        plotly_plot3D(samples.sample_stat(signal)[0].val)
        plotly_plot3D(samples.sample_stat(signal)[1].val)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    mean, var = samples.sample_stat(signal)
    plot.add(mean, title="Posterior Mean", vmin=0, vmax=1)
    plot.add(var.sqrt(), title="Posterior Standard Deviation", vmin=0)

    nsamples = samples.n_samples
    logspec = pspec.log()
    #plot.add(list(samples.iterator(pspec)) +
    #         [pspec.force(ground_truth), samples.average(logspec).exp()],
    #         title="Sampled Posterior Power Spectrum",
    #         linewidth=[1.] * nsamples + [3., 3.],
    #         label=[None] * nsamples + ['Ground truth', 'Posterior mean'])
    return mean.val, var.sqrt().val, master