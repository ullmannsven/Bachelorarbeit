from pymor.basic import *
import numpy as np
import math as m
import time 
from vkoga.vkoga import VKOGA
from vkoga.kernels import Gaussian
from ownwork import problems
from ownwork import discretizer
from itertools import count
from pymor.discretizers.builtin.cg import InterpolationOperator
from pymor.parameters.base import Mu
from matplotlib import pyplot as plt
from pymor.discretizers.builtin.cg import InterpolationOperator
import csv 
from mpl_toolkits.mplot3d import Axes3D # required for 3d plots
from matplotlib import cm # required for colors
import matplotlib.pyplot as plt
from time import perf_counter
import matplotlib as mpl
from functools import partial
from scipy.optimize import minimize

def fom_objective_functional(mu):
    return fom.output(mu)[0, 0]

problem = problems.linear_problem()
mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
fom, data = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)
parameter_space = fom.parameters.space(0, np.pi)
mu_init = [0.25, 0.5]
mu_init = problem.parameters.parse(mu_init)

diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(mu_init)
fom.visualize(diff)

mpl.rcParams['figure.figsize'] = (12.0, 8.0)
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.subplot.bottom'] = .1
mpl.rcParams['axes.facecolor'] = (0.0, 0.0, 0.0, 0.0)

def compute_value_matrix(f, x, y):
    f_of_x = np.zeros((len(x), len(y)))
    for ii in range(len(x)):
        for jj in range(len(y)):
            f_of_x[ii][jj] = f((x[ii], y[jj]))
    x, y = np.meshgrid(x, y)
    return x, y, f_of_x


def plot_3d_surface(f, x, y, alpha=1):
    X, Y = x, y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, f_of_x = compute_value_matrix(f, x, y)
    data = np.column_stack((x.reshape(-1,1), y.reshape(-1,1), f_of_x.reshape(-1,1)))
    with open('data.csv', 'w+') as f: 
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['X', 'Y', 'Z'])
        for i in range(len(data[:,0])):
            writer.writerow(data[i,:])
    #np.savetxt('data.csv',[p for p in zip(x.reshape(-1,1), y.reshape(-1,1), f_of_x.reshape(-1,1))] , delimiter=',', fmt='%s')
    np.savetxt('data.csv', data, delimiter=',', header='X,Y,Z', comments='')
    ax.plot_surface(x, y, f_of_x, cmap='Blues',
                    linewidth=0, antialiased=False, alpha=alpha)
    ax.view_init(elev=27.7597402597, azim=-39.6370967742)
    ax.set_xlim3d([-0.10457963, 3.2961723])
    ax.set_ylim3d([-0.10457963, 3.29617229])
    return ax

def addplot_xy_point_as_bar(ax, x, y, color='orange', z_range=None):
    ax.plot([y, y], [x, x], z_range if z_range else ax.get_zlim(), color)
    

ranges = parameter_space.ranges['diffusion']
XX = np.linspace(ranges[0] + 0.05, ranges[1], 10)

plot_3d_surface(fom_objective_functional, XX, XX)

def prepare_data(offline_time=False, enrichments=False):
    data = {'num_evals': 0, 'evaluations' : [], 'evaluation_points': [], 'time': np.inf}
    if offline_time:
        data['offline_time'] = offline_time
    if enrichments:
        data['enrichments'] = 0
    return data

def record_results(function, data, adaptive_enrichment=False, opt_dict=None, mu=None):
    if adaptive_enrichment:
        # this is for the adaptive case! rom is shiped via the opt_dict argument.
        assert opt_dict is not None
        QoI, data, rom = function(mu, data, opt_dict)
        opt_dict['opt_rom'] = rom
    else:
        QoI = function(mu)
    data['num_evals'] += 1
    data['evaluation_points'].append(mu)
    data['evaluations'].append(QoI)
    return QoI

def optimize(J, data, ranges, gradient=False, adaptive_enrichment=False, opt_dict=None):
    tic = perf_counter()
    result = minimize(partial(record_results, J, data, adaptive_enrichment, opt_dict),
                      [0.25, 0.5],
                      method='L-BFGS-B', jac=gradient,
                      bounds=(ranges, ranges),
                      options={'ftol': 1e-15, 'gtol': 5e-5})
    data['time'] = perf_counter()-tic
    return result

def report(result, data, reference_mu=None):
    if (result.status != 0):
        print('\n failed!')
    else:
        print('\n succeeded!')
        print(f'  mu_min:    {fom.parameters.parse(result.x)}')
        print(f'  J(mu_min): {result.fun}')
        if reference_mu is not None:
            print(f'  absolute error w.r.t. reference solution: {np.linalg.norm(result.x-reference_mu):.2e}')
        print(f'  num iterations:     {result.nit}')
        print(f'  num function calls: {data["num_evals"]}')
        print(f'  time:               {data["time"]:.5f} seconds')
        if 'offline_time' in data:
                print(f'  offline time:       {data["offline_time"]:.5f} seconds')
        if 'enrichments' in data:
                print(f'  model enrichments:  {data["enrichments"]}')
    print('')

reference_minimization_data = prepare_data()
fom_result = optimize(fom_objective_functional, reference_minimization_data, ranges)
reference_mu = fom_result.x #not used i think
report(fom_result, reference_minimization_data)

fig1 = plt.figure(1)
reference_plot = plot_3d_surface(fom_objective_functional, XX, XX, alpha=0.5)
ax = fig1.add_subplot(111)
counter = 0
X_mu = np.zeros((1,len(reference_minimization_data['evaluation_points'])))
Y_mu = np.zeros((1,len(reference_minimization_data['evaluation_points'])))
for mu in reference_minimization_data['evaluation_points']:
    addplot_xy_point_as_bar(reference_plot, mu[0], mu[1])
    X_mu[0,counter] = mu[0]
    Y_mu[0,counter] = mu[1]
    counter += 1
print(X_mu)
print()
print(Y_mu)
counter=0
for i,j in zip(X_mu, Y_mu):
    print(i,j)
    print()
    ax.annotate(str(counter), xy=(i,j), color="red")
    counter += 1

plt.show(block=True)