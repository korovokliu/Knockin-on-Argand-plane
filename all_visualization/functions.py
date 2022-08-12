from typing import List
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, pi, e, log, log2, log10
from abc import ABC, abstractmethod

avaible_functions = {
    "quadratic_complex_root": lambda x: x**2 + 2*x + 2,
    "cubic_with_complex_root": lambda x: x**3 + 2*x**2 + 2*x + 1,
    "sin_period": lambda x, period: np.sin(x*period*np.pi),
    "sin_phase": lambda x, part: sin(2 * x * pi + (pi * part) / 6),
    "sinh": lambda x: (e ** x - e ** (-x)) / 2,
    "cosh": lambda x: (e ** x + e ** (-x)) / 2,
    "ln": lambda x: log(x),
    "log2": lambda x: log2(x),
    "log10": lambda x: log10(x)
}


class GraphFOCV(plt.Figure):  # FOCV - function of complex variable

    def __init__(self, numbers: List[ComplexNumbers], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numbers = numbers
        # self.add_subplot(111)

    # def domain_coloring(self, func=avaible_functions["cubic_with_complex_root"]):
    #     """ Имплементируют раскраску по области определения """
    #     z = self.numbers[0]
    #     W = func(z.complex_plane_algebraic)
    #     cmap = 'hsv'
    #     norm = plt.Normalize(np.min(np.angle(W)), np.max(np.angle(W)))
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.pcolormesh(z.meshgrid_algebraic[0], z.meshgrid_algebraic[1], np.angle(W), cmap=cmap, norm=norm)
    #     ax.set_xlabel('Real part')
    #     ax.set_ylabel('Imaginary part')
    #     ax.set_title('Domain coloring of f(z) = sin(z)')
    #     plt.show()

    def domain_coloring(self, func=avaible_functions["cubic_with_complex_root"]):
        """ Имплементируют раскраску по области определения """
        z = self.numbers[0]
        W = func(z.complex_plane_algebraic)
        cmap = 'hsv'
        norm = plt.Normalize(np.min(np.angle(W)), np.max(np.angle(W)))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pcolormesh(z.meshgrid_algebraic[0], z.meshgrid_algebraic[1], np.angle(W), cmap=cmap, norm=norm)
        ax.set_xlabel('Real part')
        ax.set_ylabel('Imaginary part')
        ax.set_title('Domain coloring of f(z) = sin(z)')
        plt.show()


#g = GraphFOCV([ComplexNumbers(-100, -100, range_real=100, range_img=100, num_points_real=200, num_points_img=200)])
#g.domain_coloring()