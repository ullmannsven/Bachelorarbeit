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

###########################################################################################################################

import math as m
import time 
from vkoga.vkoga import VKOGA
from vkoga.kernels import Gaussian
from itertools import count
global_counter = count()

def projection_onto_range(parameter_space, mu):
    #project the parameter into the given range of the parameter space (in case it is laying outside)
    ranges = parameter_space.ranges
    mu_new = mu.copy()
    for j in range(mu.shape[0]):
        for (key, item) in ranges.items():
            range_ = ranges[key]
            for i in range(mu.shape[1]):
                if mu[j,i] < range_[0]:
                    mu_new[j,i] = range_[0] 
                if mu[j,i] > range_[1]:
                    mu_new[j,i] = range_[1]
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

def compute_RKHS_norm(fom):
    ranges = parameter_space.ranges['diffusion']
    x = np.linspace(ranges[0],ranges[1],10)
    XX, YY = np.meshgrid(x, x)
    X_train = np.zeros((2,100))
    counter = 0
    for i in range(10):
        for j in range(10):
            X_train[0,counter] = XX[0,i]
            X_train[1,counter] = YY[j,0]
            counter += 1

    target_values = np.zeros((100,1))
    for i in range(100):
        target_values[i,0] = fom.output([X_train[0,i], X_train[1,i]])[0,0]

    K = gauss_kernel_matrix(X_train, width=2)
    alpha = np.linalg.solve(K, target_values)
    return m.sqrt(alpha.T @ K @ alpha)

def gauss_kernel_matrix(data, width=2):
    n = len(data[0,:])
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = m.exp(-width*(np.linalg.norm(data[:,i] - data[:,j], 2)**2))
    return K 

def remove_similar_points(data):
    new_data = data.copy()
    idx = []

    n = len(data[:,0])
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(data[i,:] - data[j,:]) < 0.025:
                idx.append(j)

    new_data = np.delete(new_data, (idx), axis=0)

    return new_data

def remove_far_away_points(data, target_values, mu_k, TR_parameters):
    data_new = data.copy()
    target_values_new = target_values.copy()

    idx = []
    for i in range(len(data[:,0])):
        #TODO find good value for 1.2
        if np.linalg.norm(data[i,:] - mu_k[0,:]) > 2*TR_parameters["radius"]:
            idx.append(i)

    data_new = np.delete(data_new, (idx), axis=0)
    target_values_new = np.delete(target_values, (idx), axis=0)

    return data_new, target_values_new

def create_training_dataset_gradient(mu_k, X_train_old, y_train_old, TR_parameters, opt_fom_functional, kernel_model):
    if X_train_old is not None and y_train_old is not None:
        X_train_clean, y_train = remove_far_away_points(X_train_old, y_train_old, mu_k, TR_parameters)
        length_clean = len(X_train_clean[:,0])
        X_train = np.append(X_train_clean, mu_k, axis=0)
    else: 
        X_train = mu_k
        y_train = np.zeros((1,1))
        y_train[0,0] = opt_fom_functional.output(mu_k)[0, 0]
        length_clean = 1

    eps = 0.05
    e1 = np.array([[1, 0]])
    e2 = np.array([[0, 1]])
    fd_point_e1 = mu_k + eps*e1
    X_train = np.append(X_train, fd_point_e1, axis=0)
    fd_point_e2 = mu_k + eps*e2
    X_train = np.append(X_train, fd_point_e2, axis=0)
   
    X_train = projection_onto_range(parameter_space, X_train)
    X_train = remove_similar_points(X_train)
    
    num_of_points = len(X_train[:,0])
    
    for i in range(num_of_points-length_clean):
        y_train = np.append(y_train, opt_fom_functional.output(np.array([[X_train[length_clean+i,0], X_train[length_clean+i,1]]])), axis=0)
        next(global_counter)

    kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points)
    gradient = compute_gradient(kernel_model, mu_k)

    return X_train, y_train, num_of_points, gradient

def create_training_dataset(mu_k, X_train_old, y_train_old, TR_parameters, opt_fom_functional, gradient):
    X_train_clean, y_train = remove_far_away_points(X_train_old, y_train_old, mu_k, TR_parameters)
    length_clean = len(X_train_clean[:,0])
    X_train = np.append(X_train_clean, mu_k, axis=0)
    direction = -gradient
    
    for i in range(1,3):
        new_point = mu_k + i*direction
        X_train = np.append(X_train, new_point, axis=0)
        X_train = np.append(X_train, new_point + 0.2*i*np.random.randn(1,2), axis=0)
        
    X_train = projection_onto_range(parameter_space, X_train)
    X_train = remove_similar_points(X_train)

    num_of_points = len(X_train[:,0])
    for i in range(num_of_points-length_clean):
        y_train = np.append(y_train, opt_fom_functional.output(np.array([[X_train[length_clean+i,0], X_train[length_clean+i,1]]])), axis=0)
        next(global_counter)

    return X_train, y_train, num_of_points


#Computing the gradient via FD schema
def compute_gradient(kernel_model, mu_k):
    gradient = np.zeros((1,2))
    eps = 0.05
    e1 = np.array([[1, 0]])
    e2 = np.array([[0, 1]])
    gradient[0,0] = (kernel_model.predict(mu_k + eps*e1) - kernel_model.predict(mu_k))/(eps)
    gradient[0,1] = (kernel_model.predict(mu_k + eps*e2) - kernel_model.predict(mu_k))/(eps)

    return gradient

#TODO generalize the gauss kernel width
def gauss_eval(x,y):
    return m.exp(-2*(np.linalg.norm(x[0,:] - y[0,:], 2)**2))

#TODO only use the points, that where choosen by the greedy procedure
def power_function(mu, training_set):
    n = len(training_set[:,0])
    kernel_vector = np.zeros((n,1))
    for i in range(n):
        kernel_vector[i,0] = gauss_eval(mu, np.array([training_set[i,:]]))

    K = gauss_kernel_matrix(training_set.T)
    lagrange = np.linalg.pinv(K) @ kernel_vector

    interpolant = np.dot(lagrange[:,0], kernel_vector[:,0])
    power_val = m.sqrt(abs(gauss_eval(mu,mu) - interpolant))

    return power_val

def armijo_rule(kernel_model, training_set, parameter_space, TR_parameters, mu_i, Ji, direction, RKHS_norm):
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

def solve_optimization_subproblem_BFGS(kernel_model, training_set, parameter_space, mu_i, TR_parameters, RKHS_norm):
    print('\n______ starting BFGS subproblem _______')
    
    Ji = kernel_model.predict(mu_i)[0, 0]
    
    gradient = compute_gradient(kernel_model, mu_i)
    print("The gradient at point {} is {}".format(mu_i[0,:], gradient[0,:]))
    
    #Take care, this only works, because mu_i is size (1,2)
    B = np.eye(mu_i.size)
   
    i = 0
    while i < TR_parameters['max_iterations_subproblem']:
        if i>0:
            #TODO beschränkung nach oben
            if boundary_TR_criterium >= TR_parameters['beta_2']*TR_parameters['radius']:
                print('Boundary condition of TR satisfied, stopping the sub-problem solver now')
                break
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff < TR_parameters['safety_tolerance']:
                print('Subproblem converged: FOC = {}, mu_diff = {}, J_diff = {}'.format(normgrad, mu_diff, J_diff))
                break

        direction = -gradient

        mu_ip1, Jip1, boundary_TR_criterium = armijo_rule(kernel_model, training_set, parameter_space, TR_parameters, mu_i, Ji, direction, RKHS_norm)
        
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
        TR_parameters = {'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 15, 'max_iterations_subproblem':100,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                         'armijo_alpha': 1e-4, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-10,
                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 15
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 100
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 50
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1.e-4
        if 'safety_tolerance' not in TR_parameters:
            TR_parameters['safety_tolerance'] = 1e-16
        if 'FOC_tolerance' not in TR_parameters:
            TR_parameters['FOC_tolerance'] = 1e-10
        if 'beta_1' not in TR_parameters: 
            TR_parameters['beta_1'] = 0.5
        if 'beta_2' not in TR_parameters:
            TR_parameters['beta_2'] = 0.95
        if 'rho' not in TR_parameters:
            TR_parameters['rho'] = 0.75
        if 'enlarge_radius' not in TR_parameters:
            TR_parameters['enlarge_radius'] = True
        
    mu_k = TR_parameters['starting_parameter']

    #Transform the pymor object into a np.array
    mu_k = parse_parameter_inverse(mu_k)
    
    print('\n**************** Starting the offline phase, compute RKHS norm ***********\n')
    RKHS_norm = compute_RKHS_norm(opt_fom_functional)
    print('\n**************** Done computing the RKHS norm ***********\n')

    start_time = time.time()
    J_kernel_list = []
    FOCs = []
    times = []
    times_FOM = []

    mu_list = []
    mu_list.append(mu_k[0,:])

    J_FOM_list = []
    normgrad = 1e15
    point_rejected = False
    J_k = opt_fom_functional.output(mu_k)[0, 0]

    #Initializing the kernel via the VKOGA package
    width_gauss = 2
    kernel = Gaussian(ep=width_gauss)
    kernel_model = VKOGA(kernel=kernel, kernel_par=width_gauss, verbose=False, reg_par=1e-13)

    #train the kernel model for some initial data
    ticc = time.time()
    X_train, y_train, num_of_points, gradient = create_training_dataset_gradient(mu_k, None, None, TR_parameters, opt_fom_functional, kernel_model)
    X_train, y_train, num_of_points = create_training_dataset(mu_k, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
    times_FOM.append(time.time() - ticc)
    
    kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points-1)

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_k))
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

        mu_kp1, J_AGC, j, J_kp1 = solve_optimization_subproblem_BFGS(kernel_model, X_train, parameter_space, mu_k, TR_parameters, RKHS_norm)
        
        estimator_J = RKHS_norm*power_function(mu_kp1, X_train)

        if J_kp1 + estimator_J < J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[0,:]))
            
            print("\nSolving FOM for new interpolation points ...")
            ticc = time.time()
            X_train, y_train, num_of_points, gradient = create_training_dataset_gradient(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, kernel_model=kernel_model)
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
            times_FOM.append(time.time()-ticc)
            print("Updating the kernel model ...")
            
            kernel_model = kernel_model.fit(X_train, y_train, maxIter=int(num_of_points/2))

            J_FOM_list.append(opt_fom_functional.output(mu_k)[0,0])
            
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
            J_k = J_kp1

        elif J_kp1 - estimator_J > J_AGC:
            print("Rejecting the parameter mu {}".format(mu_ip1[0,:]))

            TR_parameters['radius'] *= TR_parameters['beta_1']
            print("Shrinking the TR radius to {}". TR_parameters['radius'])
            point_rejected = True
    
        else: 
            print("Accepting to check if new model is better")

            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            X_train, y_train, num_of_points, gradient = create_training_dataset_gradient(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, kernel_model=kernel_model)
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
            times_FOM.append(time.time() - tic)
            print("\nUpdating the kernel model ...\n")
            
            kernel_model = kernel_model.fit(X_train, y_train,maxIter=int(num_of_points/2))
            
            J_kp1 = kernel_model.predict(mu_kp1)[0, 0]
            #TODO this eval is not really necessary, but costly
            J_FOM_list.append(opt_fom_functional.output(mu_kp1)[0,0])
            
            print("k: {} - j: {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1[0,:]))

            if J_kp1 > J_AGC:
                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu {} and shrinking TR radius to {}".format(mu_kp1[0,:], TR_parameters['radius']))
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
    
    return mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM
    

mu_k = parameter_space.sample_randomly(1)[0]
mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM = TR_Kernel(fom, TR_parameters={'radius': 0.1, 
                        'sub_tolerance': 1e-8, 'max_iterations': 5, 'max_iterations_subproblem': 100,
                        'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                        'armijo_alpha': 1e-4, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-10,
                        'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True})

print("Der berechnete Minimierer", mu_list[-1])
print("Der tatsächliche Wert von mu", J_FOM_list[-1])
print("Der approxmierte Wert von J", J_kernel_list[-1])
print("Die benötigte Zeit (online phase) beträgt", times[-1])
print("FOM eval time", sum(times_FOM))
print("number of FOM eval", global_counter)
#plt.show(block=True)






