from typing import List

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, pi, e, log, log2, log10

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


class ComplexNumbers:
    """ Pass one variable or more """

    def __init__(self, real: float, img: float, range_real=None, range_img=None, num_points_real=None, num_points_img=None):
        self.a = real
        self.b = img

        self.range_a = self.a if not range_real else range_real
        self.range_b = self.b if not range_img else range_img

        compute = lambda x, range_x: 1 if x == range_x else int(abs(range_x - x)) + 1
        self.num_points_a = compute(self.a, self.range_a) if not num_points_real else num_points_real
        self.num_points_b = compute(self.b, self.range_b) if not num_points_img else num_points_img


    @property
    def magnitude(self):
        return np.sqrt(self.a ** 2 + self.b ** 2)

    @property
    def range_magnitude(self):
        return np.sqrt(self.range_a ** 2 + self.range_b ** 2)

    @property
    def magnitude_num_points(self):
        return int(abs(self.range_magnitude - self.magnitude)) + 1  # т.к. это просто модуль обычного числа, то можно
        # воспользоваться тем же алгоритмом для расчета оптимального числа точек как у алгебраической записи

    @staticmethod
    def calculate_arctan(x, y):  # значенияё [0, 2*pi)
        if x < 0:
            return np.arctan(y / x) + np.pi
        elif x >= 0 and y >= 0:
            return np.arctan(y / x)
        elif x >= 0 and y < 0:
            return np.arctan(y / x) + 2*np.pi
        elif x == 0 and y > 0:
            return np.pi/2
        elif x == 0 and y < 0:
            return -(np.pi/2)
        else:  # self.a == 0 and self.b == 0
            raise ZeroDivisionError


    @property
    def angle(self):
        return ComplexNumbers.calculate_arctan(self.a, self.b)

    @property
    def range_angle(self):
        return ComplexNumbers.calculate_arctan(self.range_a, self.range_b)


    @property
    def angle_nums(self):
        return 10  # int(abs(self.range_angle - self.angle)) + 1


    @property
    def meshgrid_algebraic(self):
        return np.meshgrid(np.linspace(self.a, self.range_a, self.num_points_a),
                           np.linspace(self.b, self.range_b, self.num_points_b))


    @property
    def complex_plane_algebraic(self):
        return self.meshgrid_algebraic[0] + 1j * self.meshgrid_algebraic[1]

    @property
    def meshgrid_polar(self):
        return np.meshgrid(np.linspace(self.magnitude, self.range_magnitude, self.magnitude_num_points),
                           np.linspace(self.angle, self.range_angle, self.angle_nums))




class GraphFOCV(plt.Figure):  # FOCV - function of complex variable

    def __init__(self, numbers: List[ComplexNumbers], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numbers = numbers
        # self.add_subplot(111)

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


g = GraphFOCV([ComplexNumbers(-100, -100, range_real=100, range_img=100, num_points_real=200, num_points_img=200)])
g.domain_coloring()