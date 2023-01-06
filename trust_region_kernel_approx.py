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

# initial_guess = [0.25, 0.5]

# from pymor.discretizers.builtin.cg import InterpolationOperator

# #diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(fom.parameters.parse(initial_guess))

# #TODO this is not functional, some import error appears.
# #fom.visualize(diff) 

# import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = (12.0, 8.0)
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['figure.subplot.bottom'] = .1
# mpl.rcParams['axes.facecolor'] = (0.0, 0.0, 0.0, 0.0)

# from mpl_toolkits.mplot3d import Axes3D # required for 3d plots
# from matplotlib import cm # required for colors

# from time import perf_counter

# def compute_value_matrix(f, x, y):
#     f_of_x = np.zeros((len(x), len(y)))
#     for ii in range(len(x)):
#         for jj in range(len(y)):
#             f_of_x[ii][jj] = f((x[ii], y[jj]))
#     x, y = np.meshgrid(x, y)
#     return x, y, f_of_x

# def plot_3d_surface(f, x, y, alpha=1):
#     X, Y = x, y
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     x, y, f_of_x = compute_value_matrix(f, x, y)
#     ax.plot_surface(x, y, f_of_x, cmap='Blues',
#                     linewidth=0, antialiased=False, alpha=alpha)
#     ax.view_init(elev=27.7597402597, azim=-39.6370967742)
#     ax.set_xlim3d([-0.10457963, 3.2961723])
#     ax.set_ylim3d([-0.10457963, 3.29617229])
#     return ax

# def addplot_xy_point_as_bar(ax, x, y, color='orange', z_range=None):
#     ax.plot([y, y], [x, x], z_range if z_range else ax.get_zlim(), color)

# ranges = parameter_space.ranges['diffusion']
# XX = np.linspace(ranges[0] + 0.05, ranges[1], 10)
#plot_3d_surface(fom_objective_functional, XX, XX)
###########################################################################################################################

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

#TODO check the len(item) == 0 case, when does this ever happen?
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

def gauss_kernel_matrix(data, width=2):
    n = len(data[0,:])
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = m.exp(-width*(np.linalg.norm(data[:,i] - data[:,j], 2)**2))
    return K 

#TODO the point sampling needs to be done in an efficient way
def create_training_dataset(mu_k, opt_fom_functional):
    X_train = np.append(mu_k.T, mu_k.T + 0.5*np.random.randn(2,8), axis=1)
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

#TODO vectorize, make this more efficient
#TODO only use the points, that where choosen by the greedy procedure
def power_function(mu, training_set):
    n = len(training_set[:,0])
    kernel_vector = np.zeros((n,1))
    for i in range(n):
        kernel_vector[i,0] = gauss_eval(mu, np.array([training_set[i,:]]))

    K = gauss_kernel_matrix(training_set.T)
    lagrange = np.linalg.solve(K, kernel_vector)

    sum = np.dot(lagrange[:,0], kernel_vector[:,0])
    power_val = m.sqrt(abs(gauss_eval(mu,mu) - sum))

    return power_val

def armijo_rule(kernel_model, training_set, parameter_space, TR_parameters, mu_i, Ji, direction):
    j = 0
    condition = True
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + (TR_parameters['initial_step_armijo']**j) * direction
        mu_ip1 = projection_onto_range(parameter_space, mu_ip1)

        Jip1 = kernel_model.predict(mu_ip1)[0, 0]

        power_val = power_function(mu_ip1, training_set)
        estimator_J = RKHS_norm*power_val
        
        if Jip1 <= (Ji - (TR_parameters['armijo_alpha'] / ((TR_parameters['initial_step_armijo']**j))) * (np.linalg.norm(mu_ip1 - mu_i)**2)) and abs(estimator_J / Jip1) <= TR_parameters['radius']:
            condition = False
        j += 1

    if condition:
        print("Warning: Maximum iteration for Armijo rule reached, proceeding with latest mu {}".format(mu_i))
        mu_ip1 = mu_i
        Jip1 = Ji
        estimator_J = TR_parameters['radius']*Ji
    
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
    
    Ji = kernel_model.predict(mu_i)[0, 0]
    
    gradient = compute_gradient(kernel_model, mu_i)
    print("The gradient at point {} is {}".format(mu_i[0,:], gradient[0,:]))
    
    #Take care, this only works, because mu_i is size (1,2)
    B = np.eye(mu_i.size)
   
    i = 0
    while i < TR_parameters['max_iterations_subproblem']:
        if i>0:
            #TODO fehlt hier nicht die beschränkung nach oben??
            if boundary_TR_criterium >= TR_parameters['beta_2']*TR_parameters['radius']:
                print('Boundary condition of TR satisfied, stopping the sub-problem solver now')
                break
                #print('______ ending BFGS subproblem _______\n')
                #return mu_ip1, J_AGC, i, Jip1
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff < TR_parameters['safety_tolerance']:
                print('Subproblem converged: FOC = {}, mu_diff = {}, J_diff = {}'.format(normgrad, mu_diff, J_diff))
                break

        direction = -gradient

        #TODO is this ever going to happen??
        if np.dot(direction[0,:], gradient[0,:]) > 0:
            print("Not a descend direction ... taking -gradient as direction")
            direction = -gradient 

        mu_ip1, Jip1, boundary_TR_criterium = armijo_rule(kernel_model, training_set, parameter_space, TR_parameters, mu_i, Ji, direction)
        
        #saving the General Cauchy Point 
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
        
        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)

        i += 1

    print('______ ending BFGS subproblem _______\n')

    return mu_ip1, J_AGC, i, Jip1


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

    start_time = time.time()
    J_kernel_list = []
    FOCs = []
    times = []

    mu_list = []
    mu_list.append(mu_k[0,:])

    J_FOM_list = []
    normgrad = 1e6
    model_improved = False
    point_rejected = False
    J_k_start = opt_fom_functional.output(mu_k)[0, 0]

    #Initializing the kernel via the VKOGA package
    width_gauss = 2
    kernel = Gaussian(ep=width_gauss)
    kernel_model = VKOGA(kernel=kernel, kernel_par=width_gauss, verbose=False)

    #train the kernel model for some initial data
    X_train, y_train, num_of_points = create_training_dataset(mu_k, opt_fom_functional)
    kernel_model = kernel_model.fit(X_train, y_train, maxIter=int(num_of_points/2))

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_k_start))
    print('Initial parameter {}'.format(mu_k))

    k = 0 
    while k < TR_parameters['max_iterations']:
        print("\n *********** starting iteration number {} ***********".format(k))
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < 2.22*1e-16:
                print('\n TR-radius below machine precision... stopping')
                break 
        else: 
            if (normgrad < TR_parameters['FOC_tolerance']):
                print('\n Stopping criteria fulfilled... stopping')
                break 

        mu_kp1, J_AGC, j, J_kp1 = solve_optimization_subproblem_BFGS(kernel_model, X_train, parameter_space, mu_k, TR_parameters)
        
        estimator_J = RKHS_norm*power_function(mu_kp1, X_train)

        if J_kp1 + estimator_J < J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[0,:]))
            
            print("\nSolving FOM for new interpolation points ...")
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, opt_fom_functional)

            print("Updating the kernel model ...")
            
            kernel_model = kernel_model.fit(X_train, y_train, maxIter=int(num_of_points/2))
            
            model_improved = True #TODO check, where this variable is used

            #Take care, if the create_dataset method gets changed, this might need to change as well
            J_FOM_list.append(y_train[0,0])
            
            if TR_parameters['enlarge_radius']:
                if len(J_FOM_list) > 2:
                    if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_k - J_kp1)) > TR_parameters['rho']:
                        TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                        print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

            print("k: {} - j: {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1[0,:]))

            mu_list.append(mu_kp1[0,:])
            times.append(time.time() - start_time)
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
            J_FOM_list.append(y_train[0,0])
            

            print("k: {} - j: {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1[0,:]))

            if J_kp1 > J_AGC:
                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu {} and shrinking TR radius to {}".format(mu_kp1, TR_parameters['radius']))
                point_rejected = True
                J_FOM_list.pop(-1)

            else: 
                print("Improvement good enough: Accpeting the new mu {}".format(mu_kp1[0,:]))

                mu_list.append(mu_kp1[0,:])
                times.append(time.time() - start_time)
                J_kernel_list.append(J_kp1)
                mu_k = mu_kp1

                if TR_parameters['enlarge_radius']:
                    if len(J_FOM_list) > 2:
                        #TODO understand the "k-1 != 0" statement here
                        if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_k - J_kp1)) > TR_parameters['rho']:
                            TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                            print("Enlarging the TR radius to {}".format(TR_parameters['radius']))
                J_k = J_kp1

        #TODO understand why the "if" is required here
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

#TODO this is really ugly, needs to be done in an other way!!
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
mu_list, J_FOM_list, J_kernel_list, FOCs, times = TR_Kernel(fom, TR_parameters={'radius': 0.1, 
                        'sub_tolerance': 1e-8, 'max_iterations': 20, 'max_iterations_subproblem':50,
                        'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                        'armijo_alpha': 1e-4, 'epsilon_i': 1e-8, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-16,
                        'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True, 'timings': True})

print("Minimierer", mu_list[-1])
print()
print("der tatsächliche Wert von mu", J_FOM_list[-1])
print()
print("Der approxmierte Wert von J", J_kernel_list[-1])
print()
print("die benötigte Zeit beträgt", times[-1])
print()
plt.show(block=True)

#TODO for tomorrow: 
#think about intelligient ways to sample the points 
#check all remaining stuff from the github, see if is needs for my algorithm 
#only use the points from the greedy procedure for the computation of the power function 
#vectorize/more efficient computation of the kernel matrix etc, trying to use more features of the vkoga packages
#think about all the ugly indicing etc in the kernel sampling methods 
#disable logging in pymor (see if that saves a little bit of time)





