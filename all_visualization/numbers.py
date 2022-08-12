from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
from abc import ABC, abstractmethod


class ComplexNumbers(ABC): # интерфейс

    @abstractmethod
    def convert(self):
        pass

    #region private statimethods
    @staticmethod
    def _set_start_data(axis_data: Dict[str, Union[int, float]]):
        axis_data["end"] = axis_data["start"] if axis_data["end"] is None else axis_data["end"]
        axis_data["density"] = 1 if axis_data["end"] == axis_data["start"] else int(
            abs(axis_data["end"] - axis_data["start"])) + 1

    @staticmethod
    def _update_data(old_axis_data: Dict[str, Union[int, float]], new_axis_data: Dict[str, Union[int, float]]):
        for key in new_axis_data.keys():
            try:
                old_axis_data[key] = new_axis_data[key]
            except KeyError:
                print(f"Ключа {key} не существует. \
                        \nДоступные ключи: \
                        \n'start' - значение действительной части \
                        \n'end' - конечное число для диапазона от 'start' до 'end'\
                        \n'density' - количество точек (делений) на действительной оси")

    #endregion

    @staticmethod
    def create_start_grid(data_x: Dict[str, Union[int, float]], data_y: Dict[str, Union[int, float]]):
        xpoints, ypoints = np.meshgrid(np.linspace(data_x["start"], data_x["end"], data_x["density"]),
                                       np.linspace(data_y["start"], data_y["end"], data_y["density"]))
        return (xpoints, ypoints)


    @staticmethod
    def create_result_grid(func, data_x: Dict[str, Union[int, float, List[np.ndarray]]],
                           data_y: Dict[str, Union[int, float, List[np.ndarray]]]):
        try:
            xpoints = data_x["grid_start"]
            ypoints = data_y["grid_start"]
        except KeyError:
            print("Кажется, сетка стартовых значений не создана. Убедитесь, что data_x и data_y имеют ключ 'grid_start'")
        else:
            x_result, y_result = [], []
            for row in range(len(xpoints)):  # обходим каждую строку в матрице
                x_result.append(
                    [])  # при каждом обходе строки исходной матрицы добавляем новую строку в результирующую матрицу
                y_result.append([])
                for value_in_row in range(len(xpoints[row])):
                    function_result = func(xpoints[row][value_in_row], ypoints[row][value_in_row])
                    if function_result is None:
                        x_result[row].append(0)
                        y_result[row].append(0)
                    else:
                        x_result[row].append(function_result[0])
                        y_result[row].append(function_result[1])
            return (x_result, y_result)

    @staticmethod
    def normilize_to_euclidean_norm(data_x: Dict[str, Union[int, float, List[np.ndarray]]],
                                    data_y: Dict[str, Union[int, float, List[np.ndarray]]]):

        vector_norm = lambda row, value: np.sqrt(x_result[row][value_in_row] ** 2 + y_result[row][value_in_row] ** 2)
        xnorm, ynorm, magnitude = [], [], []

        try:
            x_result = data_x["grid_result"]
            y_result = data_y["grid_result"]
        except KeyError:
            print("Кажется, сетка стартовых значений не создана. Убедитесь, что data_x и data_y имеют ключ 'grid_start'")
        else:
            for row in range(len(x_result)):
                xnorm.append([])
                ynorm.append([])
                magnitude.append([])
                for value_in_row in range(len(x_result[row])):
                    norm_for_row_val = vector_norm(row, value_in_row)
                    magnitude[row].append(norm_for_row_val)
                    if norm_for_row_val ** 2 > 0:
                        xnorm[row].append(x_result[row][value_in_row] / norm_for_row_val)
                        ynorm[row].append(y_result[row][value_in_row] / norm_for_row_val)
                    else:
                        xnorm[row].append(0)
                        ynorm[row].append(0)
            return (xnorm, ynorm, magnitude)


class AlgebraicNumber(ComplexNumbers):
    def __init__(self, real, imag, end_real=None, end_imag=None, density_real=None, density_imag=None):
        self._data_x = {"start": real, "end": end_real, "density": density_real}
        self._data_y = {"start": imag, "end": end_imag, "density": density_imag}
        ComplexNumbers._set_start_data(self._data_x)
        ComplexNumbers._set_start_data(self._data_y)

    # region property

    @property
    def data_x(self) -> Dict[str, Union[int, float]]:
        return self._data_x

    @property
    def data_y(self) -> Dict[str, Union[int, float]]:
        return self._data_y

    @data_x.setter
    def data_x(self, values_dict: Dict[str, Union[int, float]]) -> None:
        ComplexNumbers._update_data(self._data_x, values_dict)

    @data_y.setter
    def data_y(self, values_dict: Dict[str, Union[int, float]]) -> None:
        ComplexNumbers._update_data(self._data_y, values_dict)

    def __getitem__(self, part: str, key: str):
        try:
            if part == "real":
                return self._data_x[key]
            elif part == "imag":
                return self._data_y[key]
        except KeyError:
            print("Доступные значения part: 'real', 'img'\n \
                   Доступные значения key: 'start', 'end', 'density'")

    def create_start_grid(self, **kwargs):
        data_x_grid = kwargs.get("data_x", self.data_x)
        data_y_grid = kwargs.get("data_y", self.data_y)
        print(data_y_grid)
        return super().create_start_grid(data_x_grid, data_y_grid)


    #region implement convert
    @staticmethod
    def calculate_arctan_full(x, y):  # значения [0, 2*pi)
        if x < 0:
            return np.arctan(y / x) + np.pi
        elif x >= 0 and y >= 0:
            return np.arctan(y / x)
        elif x >= 0 and y < 0:
            return np.arctan(y / x) + 2 * np.pi
        elif x == 0 and y > 0:
            return np.pi / 2
        elif x == 0 and y < 0:
            return -(np.pi / 2)
        else:  # self.a == 0 and self.b == 0
            raise ZeroDivisionError

    def convert(self, angle_range="2pi"):  # convert to polar form
        magnitude = np.sqrt(self.data_x["start"]**2 + self.data_y["start"]**2)
        if angle_range == "2pi":
            angle = AlgebraicNumber.calculate_arctan_full(self.data_x["start"], self.data_y["start"])
        elif angle_range == "pi":
            angle = np.angle()
        else:
            raise Exception("Введите верный формат: 'pi' - θ ∈ [0; π],  '2pi' - θ ∈ [0; 2π)")
        return ({"magnitude": magnitude, "angle": angle})
    #endregion

    # def create_start_grid(self, data_x=self.data_x, data_y=self.data_y):
    #     super().create_start_grid(data_x, data_y)


class Graph:
    pass


class ComplexToolkit:
    """ Конечный юзер взаимодействует с этой штукой """
    def __init__(self, data: ComplexNumbers, graph: Graph) -> None:
        self.data = data
        self.graph = graph

    def vizualize(self):
        """ Нарисовать график нужного вида с комплексным числом в нужном формате """
        data_x, data_y = self.data.data_x, self.data.data_y
        x_grid_start, y_grid_start = self.data.create_start_grid(data_x=data_x, data_y=data_y)
        #x_grid_result, y_grid_result = self.data.create_result_grid(x_grid_start, y_grid_start)
        print(x_grid_start)




# import matplotlib.pyplot as plt
# def graph(self):
#     plt.figure("Имя графика", figsize=(13, 10), dpi=120)
#     # plt.style.use('dark_background')
#     # plt.style.use("seaborn-v0_8-darkgrid")
#     plt.style.use("bmh")
#
#     main_plot = plt.quiver(self.xpoints, self.ypoints, self.xnorm, self.ynorm, self.magnitude,
#                            cmap=plt.cm.jet, units='xy', headlength=3, headwidth=5,
#                            headaxislength=2.3, edgecolor='red')
#     plt.colorbar(main_plot)
#     # plt.title(PLOT_NAME + f'     (density={density})')
#     plt.xlabel('Re(z)')
#     plt.ylabel('Im(z)')
#     plt.tight_layout()
#     plt.show()
#
# def main(self):
#     self.first()
#     self.second()
#     self.third()
#     self.graph()



if __name__ == "__main__":
    context = ComplexToolkit(graph=Graph(), data=AlgebraicNumber(real=-10, imag=10))
    context.vizualize()
