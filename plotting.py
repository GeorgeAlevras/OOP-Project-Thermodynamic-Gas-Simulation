import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

temp = [50, 100, 150, 200, 250, 300, 350, 400]
pressure = [2.948, 5.774, 8.931, 11.415, 15.163, 18.249, 22.549, 26.079]

x = np.linspace(50, 400, 10000)
y = [6.5349e-2 * i for i in x]

fam = {'fontname': 'Times New Roman'}
params = {
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'font.size': 8,
    'font.family': 'serif',
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': [8, 8]
}

plt.plot(temp, pressure, 'x', color='r')
fit_phase, cov_phase = sp.polyfit(temp, pressure, 1, cov=True)
p_phase = sp.poly1d(fit_phase)
print('y-intercept: ', fit_phase[1], cov_phase[1][1])
plt.plot(temp, p_phase(temp), color='k',
         label='Fit (N:100), gradient = ' + str(round(fit_phase[0], 7)) + "+-" + str(round(cov_phase[0][0], 6)) +
         "\nb = 1.113x10$^-$$^2$$^0$ +- 0.001x10$^-$$^2$$^0$" +
         "\na = 4x10$^-$$^4$$^1$ +- 1x10$^-$$^4$$^1$")
plt.plot(x, y, color='b', label='Theoretical Van Der Waals')
plt.title("Pressure of Gas Varying Temperature", fontsize=16, **fam)
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure of Gas [mPa]")
plt.legend()
plt.savefig('pressure_successful.png')
plt.show()
