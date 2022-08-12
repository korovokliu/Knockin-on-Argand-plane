import numpy as np
from matplotlib import pyplot as plt
from all_visualization.numbers import ComplexNumbers, PolarForm


def test_polar_form():
    num_list = [ComplexNumbers(100, 100), ComplexNumbers(real=-200, range_real=200, num_points_real=400, img=500)]
    for obj in num_list:
        print(f"obj: {obj}")
        polar = PolarForm(base_real_instance=obj)



def test_complex_numbers():

    calc_magn = lambda a, b: np.sqrt(a**2 + b**2)
    calc_angle = lambda a, b: np.angle(a+b*1j)  # значения (-pi, pi]
    calc_back_to_real = lambda magn, angle: magn * np.cos(angle)
    calc_back_to_img = lambda magn, angle: magn * np.sin(angle)

    data1 = ComplexNumbers(10, -2)
    correct_meshgrid = np.meshgrid(np.linspace(10, 10, 1), np.linspace(-2, -2, 1))
    assert np.array_equal(data1.meshgrid_algebraic, correct_meshgrid)
    assert np.array_equal(data1.complex_plane_algebraic, correct_meshgrid[0]+correct_meshgrid[1]*1j)
    assert calc_magn(10, -2) == data1.magnitude
    print(data1.angle)
    print(calc_angle(10, -2))
    assert calc_angle(10, -2) == data1.angle
    assert calc_back_to_real(data1.magnitude, data1.angle) == float(data1.a)
    assert calc_back_to_img(data1.magnitude, data1.angle) == float(data1.b)

    data2 = ComplexNumbers(-4, -2, range_real=5, range_img=6)
    correct_meshgrid = np.meshgrid(np.linspace(-4, 5, 9), np.linspace(-2, 6, 8))
    assert np.array_equal(data2.meshgrid_algebraic, correct_meshgrid)
    assert np.array_equal(data2.complex_plane_algebraic, correct_meshgrid[0]+correct_meshgrid[1]*1j)
    assert calc_magn(-4, -2) == data2.magnitude
    assert calc_angle(-4, -2) == data2.angle
    assert calc_back_to_real(data2.magnitude, data1.angle) == float(data2.a)
    assert calc_back_to_img(data2.magnitude, data1.angle) == float(data2.b)


def draw_plot():
    obj = [ComplexNumbers(10, -2), ComplexNumbers(-4, -2, range_real=5, range_img=6)]
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    func1 = lambda x, y: np.sin(x) + np.cos(y)

    #for obj in objects:
    ax.pcolormesh(obj[0].meshgrid_polar[0], obj[0].meshgrid_polar[1], func1(obj[0].meshgrid_polar[0], obj[0].meshgrid_polar[1]), cmap='inferno')
#         magn_vals = np.linspace(comp_obj["magn"], comp_obj["magn"]*10, 11)
#         angle_vals = np.linspace(comp_obj["angle"], comp_obj["angle"]*2*np.pi, np.pi/12)
#         polar_meshgrid = np.meshgrid(magn_vals, angle_vals)
#
#         fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#         ax.pcolormesh(polar_meshgrid[0], polar_meshgrid[0], comp_obj["magn"]**2, cmap='inferno')
    plt.show()

def simple():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.linspace(0, 2 * np.pi, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)

    plt.pcolormesh(X, Y, Z, cmap='coolwarm')
    plt.colorbar()
    plt.show()



def simple_colorbar():
    # f(a, b) = ((a**2 - 2)*(a - (1+b))**2) / ((a+2*b)*(a**2-(5+2*b)))
    polynom = lambda a, b: ((a**2 - 2)*(a - (1+b))**2) / ((a+2*b)*(a**2-(5+2*b)))

    a = ComplexNumbers(-50, 50, range_real=100, range_img=200)
    #print(f"a.meshgrid_polar[0]: {a.meshgrid_polar[0]}")
    #print(f"a.meshgrid_polar[1]: {a.meshgrid_polar[1]}")
    Z = polynom(a.meshgrid_polar[0], a.meshgrid_polar[1])
    plt.pcolormesh(a.meshgrid_polar[0], a.meshgrid_polar[1], Z, cmap='hsv')
    plt.colorbar()
    plt.show()


def simple_colorbar2():
    # f(a, b) = ((a**2 - 2)*(a - (1+b))**2) / ((a+2*b)*(a**2-(5+2*b)))
    polynom = lambda a, b: ((a ** 2 - 2) * (a - (1 + b)) ** 2) / ((a + 2 * b) * (a ** 2 - (5 + 2 * b)))

    a = ComplexNumbers(-50, 50, range_real=100, range_img=200)
    # print(f"a.meshgrid_polar[0]: {a.meshgrid_polar[0]}")
    # print(f"a.meshgrid_polar[1]: {a.meshgrid_polar[1]}")
    Z1 = polynom(a.meshgrid_algebraic[0], a.meshgrid_algebraic[1])
    Z2 = np.sin(a.meshgrid_algebraic[0]) + np.cos(a.meshgrid_algebraic[1])
    plt.pcolormesh(a.meshgrid_algebraic[0], a.meshgrid_algebraic[1], Z2, cmap='hsv')
    plt.colorbar()
    plt.show()


def simple2():
    a = ComplexNumbers(-4, -2, range_real=5, range_img=6)
    func1 = a.meshgrid_polar[0]*(np.cos(a.meshgrid_polar[1]))
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #ax.pcolormesh(a.meshgrid_polar[0], a.meshgrid_polar[1], func1, cmap='inferno')
    ax = plt.subplot(111, projection='polar')
    ax.pcolormesh(a.meshgrid_polar[1], a.meshgrid_polar[0], func1, cmap='hsv')
    plt.show()


def simple3():
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, 1, 50)
    R, Theta = np.meshgrid(r, theta)
    Z = R * np.sin(Theta)

    ax = plt.subplot(111, projection='polar')
    ax.pcolormesh(Theta, R, Z, cmap='coolwarm')
    ax.set_title('Polar Pseudocolor Plot')
    plt.show()


if __name__ == "__main__":
    #test_complex_numbers()
    #draw_plot()
    #simple2()
    #simple_colorbar2()
    test_polar_form()