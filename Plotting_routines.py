
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def Plotting(field, DOASs, REFLs, Measurement_devices, lines, measurements, dimension, mean_rslt, std_rslt):
    
    X_field, Y_field, Z_field = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    plot_data = [go.fieldume(
        x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
        value=field.flatten(),
        name=r'NOx-Cloud',
        isomin=0.1*np.max(field),
        isomax=np.max(field),
        opacity=0.2,
        surface_count=20,
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
                                      text=["", "{}".format(round(measurements[i], 2))],textfont={"size":14},
                                      name="D{}->R{}".format(doas_ids[doas_idx], reflector_ids[reflector_idx]),
                                      marker=dict(color="rgba(0, 0, 0, 1)")))
        reflector_idx += 1
        if reflector_idx == len(reflector_ids):
            reflector_idx = 0
            doas_idx += 1
        i+=1
    
    i=0
    for DOAS_position in DOAS_positions:
        print(i, DOASs[i])
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
    
    
    Measuring_situation_plot = go.Figure(data=plot_data)
    Measuring_situation_plot.show()
    
    
    if dimension == "2D":
        mean_rslt, std_rslt = Measurement_devices.IFT8_inversion_2D()
        print(mean_rslt)
        print(std_rslt)
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
        results = [go.fieldume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=field.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                   go.fieldume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=mean_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                   go.fieldume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
                             value=field.flatten()-mean_rslt.flatten(), isomin=0.1, isomax=1., opacity=0.1, surface_count=25),
                   go.fieldume(x=X_field.flatten(), y=Y_field.flatten(), z=Z_field.flatten(),
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


