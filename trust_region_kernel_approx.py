from pymor.basic import *
import numpy as np
import matplotlib.pyplot as plt

domain = RectDomain(([-1,-1], [1,1]))
indicator_domain = ExpressionFunction(
    '(-2/3. <= x[0]) * (x[0] <= -1/3.) * (-2/3. <= x[1]) * (x[1] <= -1/3.) * 1. \
   + (-2/3. <= x[0]) * (x[0] <= -1/3.) *  (1/3. <= x[1]) * (x[1] <=  2/3.) * 1.',
    dim_domain=2)
rest_of_domain = ConstantFunction(1, 2) - indicator_domain

l = ExpressionFunction('0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', dim_domain=2)

parameters = {'diffusion': 2}
thetas = [ExpressionParameterFunctional('1.1 + sin(diffusion[0])*diffusion[1]', parameters,
                                       derivative_expressions={'diffusion': ['cos(diffusion[0])*diffusion[1]',
                                                                             'sin(diffusion[0])']}),
          ExpressionParameterFunctional('1.1 + sin(diffusion[1])', parameters,
                                       derivative_expressions={'diffusion': ['0',
                                                                             'cos(diffusion[1])']}),

                                       ]

diffusion = LincombFunction([rest_of_domain, indicator_domain], thetas)

theta_J = ExpressionParameterFunctional('1 + 1/5 * diffusion[0] + 1/5 * diffusion[1]', parameters,
                                        derivative_expressions={'diffusion': ['1/5','1/5']})

problem = StationaryProblem(domain, l, diffusion, outputs=[('l2', l * theta_J)])

mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])

fom, data = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)
parameter_space = fom.parameters.space(0, np.pi)

def fom_objective_functional(mu):
    return fom.output(mu)[0, 0]

initial_guess = [0.25, 0.5]

from pymor.discretizers.builtin.cg import InterpolationOperator

diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(fom.parameters.parse(initial_guess))

#TODO this is not functional, some import error appears.
#fom.visualize(diff) 


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.0, 8.0)
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.subplot.bottom'] = .1
mpl.rcParams['axes.facecolor'] = (0.0, 0.0, 0.0, 0.0)

from mpl_toolkits.mplot3d import Axes3D # required for 3d plots
from matplotlib import cm # required for colors

from time import perf_counter

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
print("hu", XX)
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

def optimize_BFGS(J, data, ranges, gradient=False, adaptive_enrichment=False, opt_dict=None):
    tic = perf_counter()
    result = minimize(partial(record_results, J, data, adaptive_enrichment, opt_dict),
                      initial_guess,
                      method='L-BFGS-B', jac=gradient,
                      bounds=(ranges, ranges),
                      options={'ftol': 1e-15, 'gtol': 5e-5})
    data['time'] = perf_counter()-tic
    return result

reference_minimization_data = prepare_data()
## so far everything is copy pasted from pymor

# now myown implementation for starts
from scipy.optimize import SR1
from scipy.optimize import NonlinearConstraint 
from scipy.optimize import BFGS
from scipy.optimize import minimize
import math as m
k = 0 #outer iteration index is called k, the one for the inner optimization task well be later called l
loop_flag = True

initial_guess = np.array([[0.25, 0.5]]) #initial guess for correct minimizing value, in the future multi starts might be an option
del_tr = 0.5 #initial TR radius. dont know how to chose this best, maybe try and error 
beta_1 = 0.5 #TR shringking factor
beta_2 = 0.8 #safeguard for TR boundary, usually chosen close to one
nu = 0.9 #tolerance for enlarging the TR radius
#tau_sub = 0.05 #paper proposes << 1, not sure what choice works fine here, this is the terminating condition for one of the two possible stopping conditions
tau_FOC = 0.5 #termination condition for the whole TR algorithm
width_gauss = 10 #TODO check if this yields a good approximation, otherwise *10 or /10
#TODO is it useful, to keep track of the old TR values and the minimizing sequence??

#TODO vectorize
def gauss_kernel(data, width):
    n = len(data[0,:])
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = m.exp(-width*(np.linalg.norm(data[:,i] - data[:,j], 2)**2))
    return K 

while loop_flag:
    #create a random point set close to the current iterate, which is the initial guess in the first iteration
    #TODO this is going to be the most intricate part of the algorithm, because it is unclear how to do this
    random_data_set = initial_guess.T + 0.25*np.random.randn(2,5)
    num_of_points = len(random_data_set[0,:])

    #TODO vectorize
    target_values = np.zeros((num_of_points,1))
    for i in range(num_of_points):
        target_values[i,0] = fom_objective_functional((random_data_set[0,i], random_data_set[1,i]))

    #compute kernel matrix and solver linear eq system
    K = gauss_kernel(random_data_set, width=width_gauss)
    alpha = np.linalg.solve(K, target_values)

    #comoute the kernel interpolant via lincomb of alpha_i and kernel values, TODO make this more efficient?
    def kernel_interpolant(x):
        interpol_val = 0
        for j in range(num_of_points):
            interpol_val += alpha[j,0]*m.exp(-width_gauss*(np.linalg.norm(random_data_set[:,j] - x, 2)**2))
        return interpol_val

    #TODO i dont know how to compute this, RKHS norm missing
    del_kernel_interpolant = 0
    
    # defining the nonlinear constraints of this optimization problem
    # mu is the parameter we want to optimize over, we could also write everything in term of the auxilary variable "s" and add s to the old iterate eventually
    def cons_crit(mu): 
        return [del_kernel_interpolant(mu) / kernel_interpolant(mu),np.linalg.norm(mu - initial_guess, 2)]
    nonlinear_constraint = NonlinearConstraint(cons_crit, [beta_2*del_k, 0], [del_k, del_tr], jac="2-point", hess=BFGS())

    #solving the constraint inner optimization problem
    #TODO in order to compute the generalized cauchy point, we might need the first iterate of this process
    #Keil et al refers to an algorithm in "iteratvie Methods for optimization" by C.T. Kelly, which return the seq 
    # all the inner iterates. Check if this can be used
    #TODO otherwise, we can use Def 4.1 in Keil
    #If we approximate the Jacobian (Jac) via finite differences anyway, this can be used here,
    # compute the Hessian as well is to intricate for now i guess, not to useful anyway
    res = minimize(kernel_interpolant, initial_guess, method="trust-constr", jac="2-point", hess=SR1(),
                 options={'verbose': 1},
                 constraints=nonlinear_constraint)
    print(res)
    loop_flag=False

#somehow needed if i want to run everything in the vs code terminal, stack overflow reference
plt.show(block=True)





