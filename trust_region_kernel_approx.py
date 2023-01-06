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

#diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(fom.parameters.parse(initial_guess))

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
#plot_3d_surface(fom_objective_functional, XX, XX)

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

# def optimize_BFGS(J, data, ranges, gradient=False, adaptive_enrichment=False, opt_dict=None):
#     tic = perf_counter()
#     result = minimize(partial(record_results, J, data, adaptive_enrichment, opt_dict),
#                       initial_guess,
#                       method='L-BFGS-B', jac=gradient,
#                       bounds=(ranges, ranges),
#                       options={'ftol': 1e-15, 'gtol': 5e-5})
#     data['time'] = perf_counter()-tic
#     return result

#reference_minimization_data = prepare_data()

###########################################################################################################################

## so far everything is copy pasted from pymor

# now my own implementation for starts
#from scipy.optimize import SR1
#from scipy.optimize import NonlinearConstraint 
#from scipy.optimize import BFGS
#from scipy.optimize import minimize
import math as m
import time 
import numpy as np
from vkoga.vkoga import VKOGA
from vkoga.kernels import Gaussian

def projection_onto_range(parameter_space, mu):
    #project the parameter into the given range of the parameter space (in case it is laying outside)
    ranges = parameter_space.ranges
    mu_new = mu.copy()
    for (key, item) in ranges.items():
        range_ = ranges[key]
        for i in range(mu.shape[1]):
            if mu[0,i] < range_[0]:
                mu_new[0,i] = range_[0] 
            if mu[0,i] > range_[1]:
                mu_new[0,i] = range_[1]
    return mu_new

def parse_parameter_inverse(mu):
    #convert a parameter into a numpy_array
    mu_k = []
    for (key, item) in mu._raw_values.items():
        if len(item) == 0:
            mu_k.append(mu[key][()])
        else:
            for i in range(len(item)):
                mu_k.append(mu[key][i])
    mu_array = np.array(mu_k, ndmin=2)
    return mu_array

# def pre_parse_parameter(mu_list, parameter_space):
#     # convert a list into a list that can be put into parse_parameter
#     mu_k = []
#     k = 0
#     for (key, item) in parameter_space.parameter_type.items():
#         if len(item) == 0:
#             mu_k.append(mu_list[k])
#             k+=1
#         else:
#             if item[0] == 1:
#                 mu_k.append(mu_list[k])
#                 k+=1
#             else:
#                 mu_k_add = []
#                 for i in range(item[0]):
#                     mu_k_add.append(mu_list[k])
#                     k += 1
#                 mu_k.append(mu_k_add)
#     return mu_k

def gauss_kernel_matrix(data, width=2):
    n = len(data[0,:])
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = m.exp(-width*(np.linalg.norm(data[:,i] - data[:,j], 2)**2))
    return K 

# def compute_surogate_model(mu_k):
#     #create a random point set close to the current iterate, which is the initial guess in the first iteration
#     #TODO this is going to be the most intricate part of the algorithm, because it is unclear how to do this
#     random_data_set = mu_k.T + 0.25*np.random.randn(2,5)
#     num_of_points = len(random_data_set[0,:])

#     #TODO vectorize with vkoga package
#     target_values = np.zeros((num_of_points,1))
#     for i in range(num_of_points):
#         target_values[i,0] = fom_objective_functional((random_data_set[0,i], random_data_set[1,i]))

#     #compute kernel matrix and solver linear eq system
#     K = gauss_kernel_matrix(random_data_set, width=10)
#     alpha = np.linalg.solve(K, target_values)

#     return random_data_set, num_of_points, alpha

def create_training_dataset(mu_k, opt_fom_functional):
    X_train = np.append(mu_k.T, mu_k.T + 0.5*np.random.randn(2,12), axis=1)
    #TODO mega hässlich so
    eps = 0.01
    e1 = np.array([[1, 0]])
    e2 = np.array([[0, 1]])
    new_p = mu_k.T + eps*e1.T
    new_m = mu_k.T - eps*e1.T
    X_train = np.append(X_train, new_p, axis=1)
    X_train = np.append(X_train, new_m, axis=1)

    new_p = mu_k.T + eps*e2.T
    new_m = mu_k.T - eps*e2.T
    X_train = np.append(X_train, new_p, axis=1)
    X_train = np.append(X_train, new_m, axis=1)
    X_train = projection_onto_range(parameter_space, X_train)
    
    num_of_points = len(X_train[0,:])
    y_train = np.zeros((num_of_points,1))

    #TODO should the FOM be used here??
    for i in range(num_of_points):
        y_train[i,0] = opt_fom_functional.output(np.array([[X_train[0,i], X_train[1,i]]]))[0, 0]

    return X_train.T, y_train, num_of_points

def compute_gradient(kernel_model, mu_k):
    gradient = np.zeros((1,2))
    eps = 0.01
    e1 = np.array([[1, 0]])
    e2 = np.array([[0, 1]])
    gradient[0,0] = (kernel_model.predict(mu_k + eps*e1) - kernel_model.predict(mu_k - eps*e1))/(2*eps)
    gradient[0,1] = (kernel_model.predict(mu_k + eps*e2) - kernel_model.predict(mu_k - eps*e2))/(2*eps)
    return gradient

#TODO generalize the gauss kernel width
def gauss_eval(x,y):
    return m.exp(-2*(np.linalg.norm(x[0,:] - y[0,:], 2)**2))

#TODO vectorize 
def power_function(mu, training_set):
    n = len(training_set[:,0])
    kernel_vector = np.zeros((n,1))
    for i in range(n):
        kernel_vector[i,0] = gauss_eval(mu, np.array([training_set[i,:]]))

    K = gauss_kernel_matrix(training_set.T)
    lagrange = np.linalg.solve(K, kernel_vector)

    sum = np.dot(lagrange[:,0], kernel_vector[:,0])
    #for i in range(n):
    #    sum += lagrange[i,0]*gauss_eval(mu, np.array([training_set[i,:]]))
    power_val = m.sqrt(abs(gauss_eval(mu,mu) - sum))
    return power_val

#TODO wohin muss diese funktion??
        # def kernel_interpolant(mu, random_data_set, num_of_points, alpha):
        #     interpol_val = 0
        #     for j in range(num_of_points):
        #         interpol_val += alpha[j,0]*m.exp(-width_gauss*(np.linalg.norm(random_data_set[:,j] - mu, 2)**2))
        #     return interpol_val

def armijo_rule(kernel_model, training_set, parameter_space, TR_parameters, mu_i, Ji, direction):
    j = 0
    condition = True
    #print("direction", direction)
    #print("beginning of armijo", mu_i, Ji)
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + (TR_parameters['initial_step_armijo']**j) * direction
        #print()
        #print("vorgeschlagene iterierte armijo", mu_ip1)
        mu_ip1 = projection_onto_range(parameter_space, mu_ip1)
        Jip1 = kernel_model.predict(mu_ip1)[0, 0]
        power_val = power_function(mu_ip1, training_set)
        #print("power value", power_val)
        estimator_J = RKHS_norm*power_val
        #print("neuer func value", Jip1)
        #print("VGL echter wert", fom.output(mu_ip1)[0,0])
        #print("vgl abbruch",  Ji - (TR_parameters['armijo_alpha'] / ((TR_parameters['initial_step_armijo']**j))) * (np.linalg.norm(mu_ip1 - mu_i)**2))
        #print("est / jip", abs(estimator_J / Jip1))
        #print("TR rad", TR_parameters["radius"])
        #print()
        if Jip1 <= (Ji - (TR_parameters['armijo_alpha'] / ((TR_parameters['initial_step_armijo']**j))) * (np.linalg.norm(mu_ip1 - mu_i)**2)) and abs(estimator_J / Jip1) <= TR_parameters['radius']:
            condition = False
        j += 1

    if condition:
        print("Warning: Maximum iteration for Armijo rule reached")
        mu_ip1 = mu_i
        Jip1 = Ji
        estimator_J = TR_parameters['radius']*Ji
    #print()
    #print("end of armijo", mu_ip1, Jip1)
    #print()
    #print()
    
    return mu_ip1, Jip1, abs(estimator_J/Jip1)

def compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B):
    yk = gradient - old_gradient
    yk = yk[0,:]
    sk = mu_i - old_mu
    sk = sk[0,:]
    den = np.dot(yk, sk)
    
    if den > 0:
        Hkyk = B @ yk
        coeff = np.dot(yk, Hkyk)
        HkykskT = np.outer(Hkyk, sk)
        skHkykT = np.outer(sk, Hkyk)
        skskT = np.outer(sk, sk)
        new_B = B + ((den + coeff)/(den*den) * skskT)  - (HkykskT/den) - (skHkykT/den)
    else: 
        new_B = np.eye(old_gradient.size)
    return new_B

def solve_optimization_subproblem_BFGS(kernel_model, training_set, parameter_space, mu_i, TR_parameters):
    print('\n______ starting BFGS subproblem _______')
    
    mus = []
    J_kernel_list = []
    FOCs = []

    mu_diff = []
    J_diff = []
    
    Ji = kernel_model.predict(mu_i)[0, 0]
    
    gradient = compute_gradient(kernel_model, mu_i)
    print("The gradient at point {} is{}".format(mu_i,gradient))
    normgrad = np.linalg.norm(gradient)

    #TODO check why this is necessary
    mu_i_1 = mu_i - gradient
    mu_i_1 = projection_onto_range(parameter_space, mu_i_1)

    B = np.eye(mu_i.size)

    i = 0
    while i < TR_parameters['max_iterations_subproblem']:
        if i>0:
            #TODO intuition bekommen, für das was hier passiert
            if boundary_TR_criterium >= TR_parameters['beta_2']*TR_parameters['radius']:
                print('Boundary condition of TR satisfied, stopping the sub-problem solver now')
                print('______ ending BFGS subproblem _______\n')
                return mu_ip1, J_AGC, i, Jip1, FOCs
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff < TR_parameters['safety_tolerance']:
                print('subproblem converged: FOC = {}, mu_diff = {}, J_diff = {}'.format(normgrad, mu_diff, J_diff))
                break

        direction = -gradient

        #TODO check if usefull
        if np.dot(direction[0,:], gradient[0,:]) > 0:
            print("Not a descend direction ... taking -gradient as direction")
            direction = -gradient 

        mu_ip1, Jip1, boundary_TR_criterium = armijo_rule(kernel_model, training_set, parameter_space, TR_parameters, mu_i, Ji, direction)

        if i == 0:
            J_AGC = Jip1
        
        #update the values for upcoming armijo iteration
        mu_diff = np.linalg.norm(mu_i - mu_ip1) / (np.linalg.norm(mu_i))
        J_diff = abs(Ji - Jip1) / abs(Ji)
        old_mu = mu_i.copy()
        mu_i = mu_ip1
        Ji = Jip1

        old_gradient = gradient.copy()
        gradient = compute_gradient(kernel_model, mu_i)
        
        mu_box = mu_i - gradient 
        first_order_criticity = mu_i - projection_onto_range(parameter_space, mu_box)
        normgrad = np.linalg.norm(first_order_criticity)
        
        #TODO check why this is used in Keil et al
        mu_i_1 = mu_i - gradient 
        mu_i_1 = projection_onto_range(parameter_space, mu_i_1)

        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)

        mus.append(mu_ip1)
        J_kernel_list.append(Ji)
        FOCs.append(normgrad)
        i += 1

    print("relative differences mu {} and J {}".format(mu_diff, J_diff))
    print('______ ending BFGS subproblem _______\n')


    return mu_ip1, J_AGC, i, Jip1, FOCs


def TR_Kernel(opt_fom_functional, TR_parameters=None):
    if TR_parameters is None:
        mu_k = parameter_space.sample_randomly(1)[0]
        TR_parameters = {'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 10, 'max_iterations_subproblem':400,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                         'armijo_alpha': 1e-4, 'epsilon_i': 1e-8, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-16,
                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True, 'timings': True}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 30
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 400
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 50
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1.e-4
        if 'full_order_model' not in TR_parameters:
            TR_parameters['full_order_model'] = False
        if 'printing' not in TR_parameters:
            TR_parameters['printing'] = False
        if 'epsilon_i' not in TR_parameters:
            TR_parameters['epsilon_i'] = 1e-8
        if 'Qian-Grepl' not in TR_parameters:
            TR_parameters['Qian-Grepl'] = False
        if 'safety_tolerance' not in TR_parameters:
            TR_parameters['safety_tolerance'] = 1e-16
        if 'FOC_tolerance' not in TR_parameters:
            TR_parameters['FOC_tolerance'] = TR_parameters['sub_tolerance']
        if 'beta_1' not in TR_parameters: 
            TR_parameters['beta_1'] = 0.5
        if 'beta_2' not in TR_parameters:
            TR_parameters['beta_2'] = 0.95
        if 'rho' not in TR_parameters:
            TR_parameters['rho'] = 0.75
        if 'enlarge_radius' not in TR_parameters:
            TR_parameters['enlarge_radius'] = True
        if 'timiings' not in TR_parameters:
            TR_parameters['timings'] = True
        
    mu_k = TR_parameters['starting_parameter']

    #Transform the pymor object into a np.array
    mu_k = parse_parameter_inverse(mu_k)
    mu_k = np.array([[0.25, 0.5]])

    tic = time.time()
    J_kernel_list = []
    FOCs = []
    times = []

    mu_list = []
    mu_list.append(mu_k[0,:])

    J_FOM_list = []
    normgrad = 1e6
    model_improved = False
    point_rejected = False
    J_k = opt_fom_functional.output(mu_k)[0, 0]

    #Initializing the kernel via the VKOGA package
    width_gauss = 2
    kernel = Gaussian(ep=width_gauss)
    kernel_model = VKOGA(kernel=kernel, kernel_par=width_gauss, verbose=False)

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_k))
    print('Initial parameter {}'.format(mu_k))

    k = 0 
    while k < TR_parameters['max_iterations']:
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < 2.22*1e-16:
                print('\n TR-radius below machine precision... stopping')
                break 
        else: 
            if (normgrad < TR_parameters['FOC_tolerance']):
                print('\n Stopping criteria fulfilled... stopping')
                break 

        X_train, y_train, num_of_points = create_training_dataset(mu_k, opt_fom_functional)
        kernel_model = kernel_model.fit(X_train, y_train, maxIter=int(num_of_points/2))

        mu_kp1, J_AGC, j, J_kp1, FOCs = solve_optimization_subproblem_BFGS(kernel_model, X_train, parameter_space, mu_k, TR_parameters)

        estimator_J = RKHS_norm*power_function(mu_kp1, X_train)

        if J_kp1 + estimator_J < J_AGC:
            print("Accepting the new mu {}".format(mu_kp1))

            print("\nSolving FOM for new interpolation points ...")
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, opt_fom_functional)

            print("Updating the kernel model ...")
            kernel_model = kernel_model.fit(X_train, y_train, maxIter=int(num_of_points/2))
            model_improved = True #TODO check, where this variable is used

            J_FOM_list.append(opt_fom_functional.output(mu_kp1)[0, 0])
            
            if TR_parameters['enlarge_radius']:
                if len(J_FOM_list) > 2:
                    if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_k - J_kp1)) > TR_parameters['rho']:
                        TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                        print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

            print("k: {} - j {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1))

            mu_list.append(mu_kp1[0,:])
            times.append(time.time() - tic)
            J_kernel_list.append(J_kp1)
            mu_k = mu_kp1

        elif J_kp1 - estimator_J > J_AGC:
            print("Rejecting the parameter mu {}".format(mu_ip1))

            TR_parameters['radius'] *= TR_parameters['beta_1']
            print("Shrinking the TR radius to {}". TR_parameters['radius'])
            point_rejected = True
    
        else: 
            print("Accepting to check if new model is better")

            print("\nSolving FOM for new interpolation points ...")
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, opt_fom_functional)

            print("\nUpdating the kernel model ...\n")
            kernel_model = kernel_model.fit(X_train, y_train,maxIter=int(num_of_points/2))
            model_improved = True

            J_kp1 = kernel_model.predict(mu_kp1)[0, 0]
            J_FOM_list.append(opt_fom_functional.output(mu_kp1)[0, 0])

            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))

            if J_kp1 > J_AGC:
                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu {} and shrinking TR radius to {}".format(mu_kp1, TR_parameters['radius']))
                point_rejected = True
                J_FOM_list.pop(-1)

            else: 
                print("Improvement good enough: Accpeting the new mu {}".format(mu_kp1))

                mu_list.append(mu_kp1[0,:])
                times.append(time.time() - tic)
                J_kernel_list.append(J_kp1[0, 0])
                mu_k = mu_kp1

                if TR_parameters['enlarge_radius']:
                    if len(J_FOM_list) > 2:
                        if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_k - J_kp1)) > TR_parameters['rho']:
                            TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                            print("Enlarging the TR radius to {}".format(TR_parameters['radius']))
                J_k = J_kp1

        if not point_rejected:
            gradient = compute_gradient(kernel_model, mu_k)
            mu_box = mu_k - gradient 
            first_order_criticity = mu_k - projection_onto_range(parameter_space, mu_box)
            normgrad = np.linalg.norm(first_order_criticity)
    
        FOCs.append(normgrad)    
        print("First order critical condition: {}".format(normgrad))  
        k += 1

    print("\n************************************* \n")

    if k >= TR_parameters['max_iterations']:
        print("WARNING: Maximum number of iteration for the TR algorithm reached")
    
    if TR_parameters['timings']:
        return mu_list, J_FOM_list, J_kernel_list, FOCs, times
    else: 
        return mu_list, J_FOM_list

#TODO this is really ugly
x = np.linspace(0,m.pi,10)
xx, yy = np.meshgrid(x, x)
X_train = np.zeros((2,100))
counter = 0
for i in range(10):
    for j in range(10):
        X_train[0,counter] = xx[0,i]
        X_train[1,counter] = yy[j,0]
        counter += 1

target_values = np.zeros((100,1))
for i in range(100):
    target_values[i,0] = fom.output([X_train[0,i], X_train[1,i]])[0,0]

K = gauss_kernel_matrix(X_train, width=2)
alpha = np.linalg.solve(K, target_values)
RKHS_norm = m.sqrt(alpha.T @ K @ alpha)

mu_k = parameter_space.sample_randomly(1)[0]
mu_list, J_FOM_list, J_kernel_list, FOCs, times = TR_Kernel(fom, TR_parameters={'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 10, 'max_iterations_subproblem':100,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 20, 'initial_step_armijo': 0.5, 
                         'armijo_alpha': 1e-4, 'epsilon_i': 1e-8, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-16,
                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True, 'timings': True})

print(mu_list)
print(J_FOM_list)
print(J_kernel_list)
plt.show(block=True)





