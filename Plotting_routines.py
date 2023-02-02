
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import nifty8 as ift
import os

def measuring_situation_plot_complete(field, Measurement_devices, DOASs, REFLs, measurements, lines, show=False):
    X_field, Y_field, Z_field = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    plot_data = [go.Volume(
        x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
        value=field.flatten(),
        name=r'NOx-Cloud',
        isomin=0.1*np.max(field),
        isomax=np.max(field),
        opacity=0.15,
        surface_count=15,
    )]
    
    i=0
    doas_idx = 0
    reflector_idx = 0
    doas_ids = [DOAS_device.ID for DOAS_device in DOASs]
    reflector_ids = [refl.ID for refl in REFLs]
    
    DOAS_positions, REFL_positions = Measurement_devices.return_positions()
    for X,Y,Z in lines:
        plot_data.append(go.Scatter3d(x=[X[0], X[int(len(X)/8)], X[-1]],
                                      y=[Y[0], Y[int(len(Y)/8)], Y[-1]],
                                      z=[Z[0], Z[int(len(Z)/8)], Z[-1]],
                                      mode='lines+text',
                                      text=["", "{}".format(round(measurements[i], 2))],textfont={"size":11},
                                      name="D{}->R{}".format(doas_ids[doas_idx], reflector_ids[reflector_idx]),
                                      marker=dict(color="rgba(0, 0, 0, 1)")))
        reflector_idx += 1
        if reflector_idx == len(reflector_ids):
            reflector_idx = 0
            doas_idx += 1
        i+=1
    
    i=0
    for DOAS_position in DOAS_positions:
        plot_data.append(go.Scatter3d(x=[DOAS_position[0]], y=[DOAS_position[1]], z=[DOAS_position[2]],
                                      mode='markers+text', text="DOAS {}".format(DOASs[i].ID),textfont={"size":18},
                                      marker=dict(size=12.,color="rgba(0, 0, 0, 1)", symbol="cross")))
        i+=1
    
    i=0
    for REFL_position in REFL_positions:
        plot_data.append(go.Scatter3d(x=[REFL_position[0]], y=[REFL_position[1]], z=[REFL_position[2]],
                                      mode='markers+text', text="REFL {}".format(REFLs[i].ID),textfont={"size":18},
                                      marker=dict(size=12.,color="rgba(0, 0, 0, 1)", symbol="diamond-open")))
        i+=1
    
    if show:
        Measuring_situation_plot = go.Figure(data=plot_data)
        Measuring_situation_plot.show()
    return plot_data

def measuring_situation_plot_only_paths(field, Measurement_devices, DOASs, REFLs, measurements, lines, show=False):
    X_field, Y_field, Z_field = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    plot_data=[]
    i=0
    doas_idx = 0
    reflector_idx = 0
    doas_ids = [DOAS_device.ID for DOAS_device in DOASs]
    reflector_ids = [refl.ID for refl in REFLs]
    
    DOAS_positions, REFL_positions = Measurement_devices.return_positions()
    for X,Y,Z in lines:
        plot_data.append(go.Scatter3d(x=[X[0], X[int(len(X)/8)], X[-1]],
                                      y=[Y[0], Y[int(len(Y)/8)], Y[-1]],
                                      z=[Z[0], Z[int(len(Z)/8)], Z[-1]],
                                      mode='lines',
                                      name="D{}->R{}".format(doas_ids[doas_idx], reflector_ids[reflector_idx]),
                                      marker=dict(color="rgba(0, 0, 0, 1)")))
        reflector_idx += 1
        if reflector_idx == len(reflector_ids):
            reflector_idx = 0
            doas_idx += 1
        i+=1
    
    i=0
    for DOAS_position in DOAS_positions:
        plot_data.append(go.Scatter3d(x=[DOAS_position[0]], y=[DOAS_position[1]], z=[DOAS_position[2]],
                                      mode='markers+text', text="D{}".format(DOASs[i].ID),textfont={"size":12},
                                      marker=dict(size=10.,color="rgba(0, 0, 0, 1)", symbol="cross")))
        i+=1
    
    i=0
    for REFL_position in REFL_positions:
        plot_data.append(go.Scatter3d(x=[REFL_position[0]], y=[REFL_position[1]], z=[REFL_position[2]],
                                      mode='markers+text', text="R{}".format(REFLs[i].ID),textfont={"size":12},
                                      marker=dict(size=10.,color="rgba(0, 0, 0, 1)", symbol="diamond-open")))
        i+=1
    
    if show:
        Measuring_situation_plot = go.Figure(data=plot_data)
        Measuring_situation_plot.show()
    return plot_data


def Plotting(field, DOASs, REFLs, Measurement_devices, lines, measurements, dimension, mean_rslt, std_rslt):
    plot_data = measuring_situation_plot(field, Measurement_devices, DOASs, REFLs, measurements, lines)
    X_field, Y_field, Z_field = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    if dimension == "2D":
        mean_rslt, std_rslt = Measurement_devices.IFT8_inversion_2D()
        contour = [go.Contour(z=field.T, colorscale='inferno', contours_coloring='heatmap'),
                   go.Contour(z=mean_rslt, colorscale='inferno', contours_coloring='heatmap'),
                   go.Contour(z=np.abs(field.T-np.array(mean_rslt)), colorscale='inferno', contours_coloring='heatmap'),
                   go.Contour(z=std_rslt, colorscale='inferno', contours_coloring='heatmap'),
                   ]
    
    
        contour_plot = go.Figure(data=contour)
        Measuring_situation_plot = go.Figure(data=plot_data)
    
        plot_figure = make_subplots(rows=2, cols=3, specs=[[{"type": "scene", "rowspan": 2}, {}, {}], [{}, {}, {}]],
                                    column_widths=[0.5, 0.25, 0.25],
                                    subplot_titles=("3D Measuring Situation", "Ground Truth", "Retrieved Field", "",
                                                    "Absolute Residual GrTr-RetrField", "Standard Deviation Retr. Field."))
        for t in Measuring_situation_plot.data:
            plot_figure.add_trace(t, row=1, col=1)
    
    
        plot_figure.add_trace(contour_plot.data[0], row=1, col=2)
        plot_figure.add_trace(contour_plot.data[1], row=1, col=3)
        plot_figure.add_trace(contour_plot.data[2], row=2, col=2)
        plot_figure.add_trace(contour_plot.data[3], row=2, col=3)
    
        plot_figure.update_layout(scene_xaxis_showticklabels=True,
                                  scene_yaxis_showticklabels=True,
                                  scene_zaxis_showticklabels=True,
                                  showlegend=False)
    
        plot_figure.show()
        
        
    else:
        results = [go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=field.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                   go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=mean_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                   go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=field.flatten()-mean_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                   go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=std_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25)]

        results_plot = go.Figure(data=results)
        Measuring_situation_plot = go.Figure(data=plot_data)

        plot_figure = make_subplots(rows=2, cols=3, specs=[[{"type": "scene", "rowspan": 2}, {"type": "scene"}, {"type": "scene"}],
                                                           [{}, {"type": "scene"}, {"type": "scene"}]],
                                    column_widths=[0.5, 0.25, 0.25],
                                    subplot_titles=("3D Measuring Situation", "Ground Truth", "Retrieved Field", "",
                                                    "Absolute Residual GrTr-RetrField", "Standard Deviation Retr. Field."))
        for t in Measuring_situation_plot.data:
            plot_figure.add_trace(t, row=1, col=1)

        plot_figure.add_trace(results_plot.data[0], row=1, col=2)
        plot_figure.add_trace(results_plot.data[1], row=1, col=3)
        plot_figure.add_trace(results_plot.data[2], row=2, col=2)
        plot_figure.add_trace(results_plot.data[3], row=2, col=3)

        plot_figure.update_layout(scene_xaxis_showticklabels=True,
                                  scene_yaxis_showticklabels=True,
                                  scene_zaxis_showticklabels=True,
                                  showlegend=False)

        plot_figure.show()

def post_analysis(field, DOASs, REFLs, MDevices, lines, measurements, nifty_mean, nifty_std, basis_field, xi_basis, target_folder):
    
    
    data = MDevices.measure()[0]
    MDevices.change_field(nifty_mean)
    nifty_data = MDevices.measure()[0]
    nifty_xi_sq = np.sum((data - nifty_data)**2)
    
    
    # --- 3D Maps ---
    diff_nifty_3D = field - nifty_mean
    diff_basis_3D = field - basis_field
    
    perc_nifty_3D = (np.abs(nifty_mean-field)/field*100).mean()
    perc_basis_3D = (np.abs(basis_field-field)/field*100).mean()
    
    corr_nifty_mean = np.corrcoef(field.flatten(), nifty_mean.flatten())[0,1]
    corr_basis_mean = np.corrcoef(field.flatten(), basis_field.flatten())[0,1]
    corr_std = np.corrcoef(np.abs(diff_nifty_3D).flatten(), nifty_std.flatten())[0,1]
    
    # --- 2D Maps ---
    av_diff_nifty = diff_nifty_3D.mean(axis=2)
    av_diff_basis = diff_basis_3D.mean(axis=2)
    
    names = ["<b>xi^2<b>", "<b>av |diff| [ppb]<b>", "<b>av |diff| [%]<b>", "<b>Correlation<b>", "<b>STD corr to diff<b>"]
    values_nifty = [round(nifty_xi_sq,5), round(np.abs(diff_nifty_3D).mean(),3), round(perc_nifty_3D,3), round(corr_nifty_mean,3), round(corr_std,3)]
    values_basis = [round(xi_basis[-1],5), round(np.abs(diff_basis_3D).mean(),3), round(perc_basis_3D,3), round(corr_basis_mean,3), "---"]
    res_table = dict(values=[names, values_nifty, values_basis], align = ['left', 'center'])
    
    plot_figure = make_subplots(rows=3, cols=3, specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                                                       [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                                                       [{"type": "table"}, {"type": "scene"}, {"type": "xy"}]],
                                    column_widths=[0.33, 0.33, 0.33],
                                    subplot_titles=("Ground Truth", "NIFTy Mean Field", "BasisFunc Field", "Setup",
                                                    "GT - NIFTY Mean", "GT - BasisFunc Field", "Reconstruction Analysis", "NIFTY Std",
                                                    "BasisFunc xi^2"),
                                    horizontal_spacing = 0.02, vertical_spacing=0.05)
    
    X_field, Y_field, Z_field = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    Measuring_situation_plot = go.Figure(data=measuring_situation_plot_only_paths(np.zeros(field.shape), MDevices, DOASs, REFLs, measurements, lines, show=False))
    
    for t in Measuring_situation_plot.data:
        plot_figure.add_trace(t, row=2, col=1)
    
    results = [go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(), colorbar=dict(len=0.3, y=0.84, x=0.25),
                        value=field.flatten(), isomin=0.1*np.max(field), isomax=np.max(field), opacity=0.2, surface_count=15),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(), colorbar=dict(len=0.3, y=0.84, x=0.6),
                        value=nifty_mean.flatten(), isomin=0.1*np.max(field), isomax=np.max(field), opacity=0.2, surface_count=15),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(), colorbar=dict(len=0.3, y=0.49, x=0.6),
                        value=diff_nifty_3D.flatten(), opacity=0.15, surface_count=15),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(), colorbar=dict(len=0.3, y=0.84, x=0.95),
                        value=basis_field.flatten(), isomin=0.1*np.max(field), isomax=np.max(field), opacity=0.2, surface_count=15),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(), colorbar=dict(len=0.3, y=0.49, x=0.95),
                        value=diff_basis_3D.flatten(), opacity=0.15, surface_count=15),
                go.Volume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(), colorbar=dict(len=0.3, y=0.12, x=0.6),
                        value=nifty_std.flatten(), isomin=0.1*np.max(nifty_std), opacity=0.15, surface_count=15),
                go.Scatter(x=np.linspace(0,xi_basis.shape[0],xi_basis.shape[0]), y=xi_basis, mode="lines"),
                go.Table(header=dict(values=['<b>Type<b>', '<b>NIFTy<b>', '<b>BasisFunctions<b>']), cells=res_table)
                ]

    results_plot = go.Figure(data=results, layout=go.Layout(autosize=False, width=3000, height=2000))
    
    plot_figure.add_trace(results_plot.data[0], row=1, col=1)
    plot_figure.add_trace(results_plot.data[1], row=1, col=2)
    plot_figure.add_trace(results_plot.data[2], row=2, col=2)
    plot_figure.add_trace(results_plot.data[3], row=1, col=3)
    plot_figure.add_trace(results_plot.data[4], row=2, col=3)
    plot_figure.add_trace(results_plot.data[5], row=3, col=2)
    plot_figure.add_trace(results_plot.data[6], row=3, col=3)
    plot_figure.add_trace(results_plot.data[7], row=3, col=1)
    
    plot_figure.update_layout(scene_xaxis_showticklabels=True,
                              scene_yaxis_showticklabels=True,
                              scene_zaxis_showticklabels=True,
                              showlegend=False)

    plot_figure.update_layout(shapes=[
        go.layout.Shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=-.05,
        x1=1,
        y1=1.05,
        line={"width": 1, "color": "black"}),
        # line
        go.layout.Shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0.33,
        y0=-0.05,
        x1=0.33,
        y1=1.05,
        line={"width": 1, "color": "black"}), 
        go.layout.Shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0.66,
        y0=-0.05,
        x1=0.66,
        y1=1.05,
        line={"width": 1, "color": "black"}), 
        go.layout.Shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0.33,
        x1=1,
        y1=0.33,
        line={"width": 1, "color": "black"}),
        go.layout.Shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0.68,
        x1=1,
        y1=0.68,
        line={"width": 1, "color": "black"})
        ],autosize=False, width=2000, height=1300)
    
    plot_figure.write_image(os.path.join(target_folder, "result.pdf"))
    return
