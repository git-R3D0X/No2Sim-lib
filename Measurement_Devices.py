# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from Line import Line, Line_2D
import nifty8 as ift


class Measurement_Devices:
    """
    contains class methods for the DOAS devices:
    -> measure
    -> Add Doas
    """

    def __init__(self, DOAS_devices, Reflectors, simulated_field):
        self.DOAS_devices = DOAS_devices
        self.Reflectors = Reflectors
        self.simulated_field = simulated_field
        if len(self.simulated_field.shape) == 2:
            self.dim = 2
        else:
            self.dim = 3
        print("Measurement Devices: \"The Dimension of the field is {}\"".format(self.dim))

    def print_parameters(self):
        for DOAS_device in self.DOAS_devices:
            DOAS_device.print_parameters()
        for Reflector in self.Reflectors:
            Reflector.print_parameters()

    def return_plottables(self):
        return_arr = []
        for DOAS_device in self.DOAS_devices:
            for Reflector in self.Reflectors:
                if self.dim == 2:
                    line_points = Line_2D(Reflector.position-DOAS_device.position, DOAS_device.position).get_plot_points()
                else:
                    line_points = Line(Reflector.position-DOAS_device.position, DOAS_device.position).get_plot_points()
                return_arr.append(line_points)
        return return_arr

    def return_positions(self):
        doas_positions = []
        for DOAS_device in self.DOAS_devices:
            doas_positions.append(DOAS_device.position)
        refl_positions = []
        for Reflector in self.Reflectors:
            refl_positions.append(Reflector.position)
        if self.dim == 3:
            return doas_positions, refl_positions
        else:
            return [k[:2] for k in doas_positions], [k[:2] for k in refl_positions]

    def measure(self):
        def build_LOSs(doas_pos, refl_pos):
            if self.dim == 2:
                LOS_starts = [[], []]
                LOS_ends = [[], []]
                for doas in doas_pos:
                    for i in range(len(refl_pos)):
                        LOS_starts[0].append(doas[0] / self.simulated_field.shape[0])
                        LOS_starts[1].append(doas[1] / self.simulated_field.shape[1])
                        LOS_ends[0].append(refl_pos[i][0] / self.simulated_field.shape[0])
                        LOS_ends[1].append(refl_pos[i][1] / self.simulated_field.shape[1])
                return LOS_starts, LOS_ends
            else:

                LOS_starts = [[], [], []]
                LOS_ends = [[], [], []]
                for doas in doas_pos:
                    for i in range(len(refl_pos)):
                        LOS_starts[0].append(doas[0] / self.simulated_field.shape[0])
                        LOS_starts[1].append(doas[1] / self.simulated_field.shape[1])
                        LOS_starts[2].append(doas[2] / self.simulated_field.shape[2])
                        LOS_ends[0].append(refl_pos[i][0] / self.simulated_field.shape[0])
                        LOS_ends[1].append(refl_pos[i][1] / self.simulated_field.shape[1])
                        LOS_ends[2].append(refl_pos[i][2] / self.simulated_field.shape[2])
                return LOS_starts, LOS_ends

        position_space = ift.RGSpace(self.simulated_field.shape)
        DOAS_positions, REFL_positions = self.return_positions()
        LOS_starts, LOS_ends = build_LOSs(DOAS_positions, REFL_positions)
        R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        ground_truth = ift.makeField(position_space, self.simulated_field)
        data = R(ground_truth)

        #plot = ift.Plot()
        #plot.add(R.adjoint_times(data), title='Data')
        #plot.output(ny=1, nx=3, xsize=24, ysize=6, name="data.png")

        return data.val, LOS_starts, LOS_ends

