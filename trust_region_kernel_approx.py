from pymor.basic import *
import numpy as np
import math as m
import time 
from vkoga.vkoga import VKOGA
from vkoga.kernels import Gaussian
import problems
import discretizer
from itertools import count
from pymor.discretizers.builtin.cg import InterpolationOperator
global_counter = count()

def fom_compute_output(mu):
    #return fom.output(mu)[0, 0]
    mu = problem.parameters.parse(mu)
    return fom.output_functional_hat(mu)

def projection_onto_range(parameter_space, mu):
    """Projects the list of parameters |mu| onto the given range of the parameter space

    Parameters
    ----------
    parameter_space
        The |parameter_space| of the full order model which is optimized
    mu
        List of parameters |mu| that is projected onto the given range

    Returns
    -------
    mu_new 
        The projected parameter list

    """
    mu_new = mu.copy()
    for j in range(len(mu[:,0])):
        index = 0
        for (key, val) in parameter_space.parameters.items():
            range_ = parameter_space.ranges[key]
            for i in range(index, index + val):
                if mu[j,i] < range_[0]:
                    mu_new[j,i] = range_[0] 
                if mu[j,i] > range_[1]:
                    mu_new[j,i] = range_[1]
                index += 1
    return mu_new

def parse_parameter_inverse(mu):
    """Transform a pymor Mu Object |mu| to a numpy array |mu_array|

    Parameters
    ----------
    mu
        The parameter |mu| that needs to be transformed to a numpy array

    Returns
    -------
    mu_array
        The numpy array with the same values

    """
    mu_k = []
    for (key, item) in mu._raw_values.items():
        if len(item) == 0:
            mu_k.append(mu[key][()])
        else:
            for i in range(len(item)):
                mu_k.append(mu[key][i])
    mu_array = np.array(mu_k, ndmin=2)
    return mu_array

def compute_RKHS_norm(fom, kernel):
    """Computes the RKHS norm of the FOM that gets to be optimized

    Parameters
    ----------
    fom
        The Full Order Model (FOM) that gets optimized
    kernel
        The reproducing kernel of the RKHS

    Returns
    -------
    rkhs_norm
        An approximation |rkhs_norm| of the RKHS norm of the FOM
    """
    amount = 30
    parameter_dim = 0
    for (key, val) in parameter_space.parameters.items():
        parameter_dim += val
    
    X_train = np.zeros((amount,parameter_dim))
    target_values = np.zeros((amount,1))
    for i in range(amount):
        mu = parameter_space.sample_randomly(1, seed=i)[0]
        mu_as_array = parse_parameter_inverse(mu)
        X_train[i,:] = mu_as_array[0,:]
        target_values[i,0] = fom_compute_output(mu)
    
    K = kernel.eval(X_train, X_train)
    alpha = np.linalg.solve(K, target_values)
    rkhs_norm = m.sqrt(alpha.T @ K @ alpha)
    return rkhs_norm

#TODO make eps as a variable
def compute_gradient(kernel_model, mu_k):
    """Approximates the gradient at the parameter |mu_k| using a forward FD scheme.

    Parameters
    ----------
    kernel_model
        The |kernel_model| which is used for approximating the Full Order Model
    mu_k 
        The parameter |mu_k| where the gradient is computed

    Returns
    -------
    gradient
        The |gradient| at parameter |mu_k|
    """
    dimension = len(mu_k[0,:])
    gradient = np.zeros((1,dimension))
    eps = 0.05
    for j in range(dimension):
        unit_vec = np.zeros((1,dimension))
        unit_vec[0,j] = 1
        gradient[0,j] = (kernel_model.predict(mu_k + eps*unit_vec) - kernel_model.predict(mu_k - eps*unit_vec))/(2*eps)
    
    return gradient

def remove_similar_points(X_train):
    """Removes points from the parameter training set |X_train| if they are to close to each other. 

    This method avoids that the resulting kernel matrix of the training set |X_train| is getting singular.

    Parameters
    ----------
    X_train
        The training set which is getting reduced

    Returns
    -------
    X_train_clean
        The cleared training set |X_train_clean|
    """
    idx = []
    num_of_points = len(X_train[:,0])
    for i in range(num_of_points):
        for j in range(i+1,num_of_points):
            if np.linalg.norm(X_train[i,:] - X_train[j,:]) < 0.049:
                idx.append(j)

    X_train_clean = np.delete(X_train, (idx), axis=0)

    return X_train_clean

def remove_far_away_points(X_train, target_values, mu_k, TR_parameters):
    """Removes points from the parameter training set |X_train| if they far away from the current iterate |mu_k|. 

    Points that are far away from the current iterate |mu_k| are often not useful for the approximation.

    Parameters
    ----------
    X_train
        The training set that gets cleaned
    target_values
        The|target_values| of the training set |X_train|
    mu_k
        The current iterate
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm

    Returns
    -------
    X_train_new 
        The cleaned data set |X_train_new|
    target_values_new
        The target_values of the cleaned training set |X_train_new|
    """
    idx = []
    for i in range(len(X_train[:,0])):
        if np.linalg.norm(X_train[i,:] - mu_k[0,:]) > TR_parameters["radius"]:
            idx.append(i)

    X_train_new = np.delete(X_train, (idx), axis=0)
    target_values_new = np.delete(target_values, (idx), axis=0)

    return X_train_new, target_values_new

def create_training_dataset_gradient(mu_k, X_train_old, y_train_old, TR_parameters, opt_fom_functional, kernel_model):
    """Adds the points, that are necessary to approximate the gradient, to the training set. 

    This method also removes points that are too far away from the current iterate |mu_k| using the :meth:remove_far_away_points

    Parameters
    ----------
    mu_k
        The current iterate |mu_k|
    X_train_old
        The training set from the last iteration
    y_train_old
        The target values corresponding to the old training set |X_train_old|
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    opt_fom_functional
        The FOM |opt_fom_functional| that gets optimized
    kernel_model
        The |kernel_model| which is used to approximate the FOM

    Returns
    -------
    X_train
        An updated training set
    y_train
        The target values corresponding to the updated training set |y_train|
    gradient
        An approxmiation of the gradient at the parameter |mu_k|
    """
    if X_train_old is not None and y_train_old is not None:
        X_train, y_train = remove_far_away_points(X_train_old, y_train_old, mu_k, TR_parameters)
        X_train = np.append(X_train, mu_k, axis=0)
        old_len = len(X_train[:,0])
        X_train = remove_similar_points(X_train)
        if old_len == len(X_train[:,0]):
            new_target_value = fom_compute_output(X_train[-1,:])
            y_train = np.append(y_train, np.array(new_target_value, ndmin=2, copy=False), axis=0)
        length_clean = len(X_train[:,0])
    else: 
        X_train = mu_k
        y_train = np.zeros((1,1))
        y_train[0,0] = fom_compute_output(mu_k)
        length_clean = 1

    dimension = len(mu_k[0,:])
    eps = 0.05
    for j in range(dimension):
        unit_vec = np.zeros((1,dimension))
        unit_vec[0,j] = 1
        fd_point_p = mu_k + eps*unit_vec
        fd_point_m = mu_k - eps*unit_vec
        X_train = np.append(X_train, fd_point_p, axis=0)
        X_train = np.append(X_train, fd_point_m, axis=0)

    X_train = projection_onto_range(parameter_space, X_train)
    X_train = remove_similar_points(X_train)
    num_of_points = len(X_train[:,0])
    
    for i in range(num_of_points-length_clean):
        new_target_value = fom_compute_output(X_train[length_clean+i,:])
        y_train = np.append(y_train, np.array(new_target_value, ndmin=2, copy=False), axis=0)
        next(global_counter)
    
    kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points)
    gradient = compute_gradient(kernel_model, mu_k)

    return X_train, y_train, gradient

def create_training_dataset(mu_k, X_train_old, y_train_old, TR_parameters, opt_fom_functional, gradient):
    """Adds new points to the training data set |X_train_old| that are lying in the direction of the gradient.

    Parameters
    ----------
    mu_k
        The current iterate |mu_k|.
    X_train_old
        The training set from the last iteration.
    y_train_old
        The target values corresponding to the old training set |X_train_old|.
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm.
    opt_fom_functional
        The FOM |opt_fom_functional| which gets optimized.
    gradient 
        An approximation of the |gradient| at the current iterate |mu_k|.

    Returns
    -------
    X_train
        An updated training set.
    y_train
        The target values corresponding to the updated training set |y_train|.
    """
    X_train, y_train = remove_far_away_points(X_train_old, y_train_old, mu_k, TR_parameters)
    length_clean = len(X_train[:,0])
    X_train = np.append(X_train, mu_k, axis=0)
    direction = -gradient
    
    new_point = mu_k + 2*direction
    X_train = np.append(X_train, new_point, axis=0)
    
    X_train = projection_onto_range(parameter_space, X_train)
    X_train = remove_similar_points(X_train)

    num_of_points = len(X_train[:,0])
    for i in range(num_of_points-length_clean):
        new_target_value = fom_compute_output(X_train[length_clean+i,:])
        y_train = np.append(y_train, np.array(new_target_value, ndmin=2, copy=False), axis=0)
        next(global_counter)

    return X_train, y_train

#TODO only use the points, that where choosen by the greedy procedure
def power_function(mu, X_train, kernel):
    """Computes the value of the Power Function for the paramter |mu|.

    Parameters
    ----------
    mu
        The parameter |mu| for which the Power function should be evaluated
    X_train
        The training set of the kernel model
    kernel
        The kernel which is used for approximating the FOM

    Returns
    -------
    power_val
        The value of the Power Function at parameter |mu|
    """
    kernel_vector = kernel.eval(X_train, mu)
    K = kernel.eval(X_train, X_train)

    lagrange_basis = np.linalg.pinv(K) @ kernel_vector
    interpolant = np.dot(lagrange_basis[:,0], kernel_vector[:,0])
    power_val = m.sqrt(abs(kernel.eval(mu,mu) - interpolant))

    return power_val

def armijo_rule(kernel_model, X_train, TR_parameters, mu_i, Ji, direction, RKHS_norm):
    """Computes a new iterate |mu_ip1| s.t it satisfies the armijo conditions.

    Parameters
    ----------
    kernel_model 
        The kernel_model that is used to approximate the FOM
    X_train 
        The training set of the kernel model
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    mu_i 
        The current iterate of the BFGS subproblem
    Ji
        The value of the functional which is optimized at parameter |mu_i|
    direction
        The negative gradient at parameter |mu_i|
    RKHS_norm 
        The approximation of the |RKHS_norm| of the FOM

    Returns
    -------
    mu_ip1
        The new parameter that satisfies the armijo conditions 
    Jip1
        The value of the functional which is optimized at parameter |mu_ip1|
    boundary_TR_criterium
        Estimates if the proposed point of the arimjo routine is already good enough to be accepted by BFGS
    """
    j = 0
    condition = True
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + (TR_parameters['initial_step_armijo']**j) * direction
        mu_ip1 = projection_onto_range(parameter_space, mu_ip1)

        Jip1 = kernel_model.predict(mu_ip1)[0, 0]

        power_val = power_function(mu_ip1, X_train, kernel_model.kernel)
        estimator_J = RKHS_norm*power_val
        
        if Jip1 <= (Ji - (TR_parameters['armijo_alpha'] / ((TR_parameters['initial_step_armijo']**j))) * (np.linalg.norm(mu_ip1 - mu_i)**2)) and abs(estimator_J / Jip1) <= TR_parameters['radius']:
            condition = False
        j += 1

    if condition:
        print("Warning: Maximum iteration for Armijo rule reached, proceeding with latest mu {}".format(mu_i))
        mu_ip1 = mu_i
        Jip1 = Ji
        estimator_J = TR_parameters['radius']*Ji
    
    boundary_TR_criterium = abs(estimator_J/Jip1)
    return mu_ip1, Jip1, boundary_TR_criterium

def compute_new_hessian_approximation(mu_i, mu_old, gradient, gradient_old, B_old):
    """Computes an approximation of the Hessian at parameter |mu_i|.

    Parameters
    ----------
    mu_i 
        The new iterate of the BFGS subproblem 
    mu_old
        The prevoius iterate of the BFGS subproblem
    gradient 
        The gradient at parameter |mu_i|
    gradient _old
        The gradient at parameter |old_mu|
    B_old
        An approximation of the Hessian at parameter |mu_old|

    Returns
    -------
    B_new
        An approximation of the Hessian at parameter |mu_i|
    """
    yk = gradient - gradient_old
    yk = yk[0,:]
    sk = mu_i - mu_old
    sk = sk[0,:]
    den = np.dot(yk, sk)
    
    if den > 0:
        Hkyk = B_old @ yk
        coeff = np.dot(yk, Hkyk)
        HkykskT = np.outer(Hkyk, sk)
        skHkykT = np.outer(sk, Hkyk)
        skskT = np.outer(sk, sk)
        B_new = B_old + ((den + coeff)/(den*den) * skskT)  - (HkykskT/den) - (skHkykT/den)
    else: 
        B_new = np.eye(gradient_old.size)

    return B_new

def optimization_subproblem_BFGS(kernel_model, training_set, mu_i, TR_parameters, RKHS_norm):
    """Solves the optimization subproblem of the TR algorithm using a BFGS with constraints.

    Parameters
    ----------
    kernel_model
        The kernel model which is used to approximate the FOM
    X_train
        The training set of the |kernel_model|
    mu_i
        The current iterate of the TR algorithm 
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    RKHS_norm 
        The approximation of the |RKHS_norm| of the FOM

    Returns
    -------
    mu_ip1
        The new iterate for the TR algorithm
    J_AGC
        The value of the functional which gets optimized at the generalized cauchy point, which is the first iterate of the subproblem
    i
        The number of iterations the subproblem needed to converge 
    Jip1
        The value of the functional which gets optimized at parameter |mu_ip1|
    """
    print('\n______ starting BFGS subproblem _______')
    
    Ji = kernel_model.predict(mu_i)[0, 0]
    
    gradient = compute_gradient(kernel_model, mu_i)
    print("The gradient at point {} is {}".format(mu_i[0,:], gradient[0,:]))
    
    B = np.eye(mu_i.size)
   
    i = 1
    while i <= TR_parameters['max_iterations_subproblem']:
        if i>1:
            #TODO beschränkung nach oben
            if boundary_TR_criterium >= TR_parameters['beta_2']*TR_parameters['radius']:
                print('Boundary condition of TR satisfied, stopping the sub-problem solver now')
                break
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff < TR_parameters['safety_tolerance']:
                print('Subproblem converged: FOC = {}, mu_diff = {}, J_diff = {}'.format(normgrad, mu_diff, J_diff))
                break

        direction = -gradient

        mu_ip1, Jip1, boundary_TR_criterium = armijo_rule(kernel_model, training_set, TR_parameters, mu_i, Ji, direction, RKHS_norm)
        
        if i == 0:
            J_AGC = Jip1
        
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
    """The Trust Region kernel algorithm which is to find the minimum of the |opt_fom_functional|.

    Parameters
    ----------
    opt_fom_functional
        The FOM which gets optimized 
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    
    Returns
    -------
    mu_list
        A list containing all iterates of the algorithm.
    J_FOM_list
        A list containing all values of the |opt_fom_functional| at the parameters contained in |mu_list|.
    J_kernel_list
        A list containing all approximated values by the kernel model at the parameters contained in |mu_list|.
    FOCs
        A list containing all FOCs.
    times
        A list containing the times measured from the beginning of the algorithm until a new sufficient iterate for the TR algorithm is computed.
    times_FOM
        A list containing the times it took to evaluate the |opt_fom_functional|.
    """
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
            TR_parameters['armijo_alpha'] = 1e-4
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
    mu_k = parse_parameter_inverse(mu_k)
    
    J_FOM_list = []
    J_kernel_list = []
    FOCs = []
    times = []
    times_FOM = []

    mu_list = []
    mu_list.append(mu_k[0,:])

    normgrad = np.inf
    point_rejected = False

    width_gauss = 2
    kernel = Gaussian(ep=width_gauss)
    kernel_model = VKOGA(kernel=kernel, kernel_par=width_gauss, verbose=False, reg_par=1e-13)

    print('\n**************** Starting the offline phase, compute RKHS norm ***********\n')
    RKHS_norm = compute_RKHS_norm(opt_fom_functional, kernel)
    print('\n**************** Done computing the RKHS norm ***********\n')

    start_time = time.time()

    tic = time.time()
    X_train, y_train, gradient = create_training_dataset_gradient(mu_k, None, None, TR_parameters, opt_fom_functional, kernel_model)
    X_train, y_train = create_training_dataset(mu_k, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
    times_FOM.append(time.time() - tic)

    X_train = projection_onto_range(parameter_space, X_train)
    X_train = remove_similar_points(X_train)
    num_of_points = len(X_train[:,0])
    kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points)

    tic = time.time()
    J_k = fom_compute_output(mu_k)
    times_FOM.append(time.time()-tic)
    J_FOM_list.append(J_k)

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_k))
    print('Initial parameter {}'.format(mu_k))

    k = 1 
    while k <= TR_parameters['max_iterations']:
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

        mu_kp1, J_AGC, j, J_kp1 = optimization_subproblem_BFGS(kernel_model, X_train, mu_k, TR_parameters, RKHS_norm)
        
        estimator_J = RKHS_norm*power_function(mu_kp1, X_train, kernel)

        if J_kp1 + estimator_J < J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[0,:]))
            
            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            X_train, y_train, gradient = create_training_dataset_gradient(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, kernel_model=kernel_model)
            X_train, y_train = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
            times_FOM.append(time.time()-tic)
            
            X_train = projection_onto_range(parameter_space, X_train)
            X_train = remove_similar_points(X_train)
            num_of_points = len(X_train[:,0])

            print("Updating the kernel model ...\n")
            kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points)

            tic = time.time()
            J_FOM_list.append(fom_compute_output(mu_kp1))
            times_FOM.append(time.time()-tic)
            
            # index = 0
            # for i in range(num_of_points):
            #     if np.linalg.norm(X_train[i,:] - mu_kp1[0,:] ) < 0.025:
            #         index = i 
            #         continue
            # J_FOM_list.append(y_train[index,0])
            
            if TR_parameters['enlarge_radius']:
                if len(J_FOM_list) > 2:
                    if abs(J_k - J_kp1) > TR_parameters['safety_tolerance']:
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
            times.append(time.time() - start_time)
    
        else: 
            print("Accepting to check if new model is better")

            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            X_train, y_train, gradient = create_training_dataset_gradient(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, kernel_model=kernel_model)
            X_train, y_train = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
            times_FOM.append(time.time()-tic)
            
            X_train = projection_onto_range(parameter_space, X_train)
            X_train = remove_similar_points(X_train)
            num_of_points = len(X_train[:,0])

            print("\nUpdating the kernel model ...\n")
            kernel_model = kernel_model.fit(X_train, y_train,maxIter=num_of_points)
            
            J_kp1 = kernel_model.predict(mu_kp1)[0, 0]

            tic = time.time()
            J_FOM_list.append(fom_compute_output(mu_kp1))
            times_FOM.append(time.time()-tic)
            # index = 0
            # for i in range(num_of_points):
            #     if np.linalg.norm(X_train[i,:] - mu_kp1[0,:] ) < 0.025:
            #         index = i 
            #         continue
            # J_FOM_list.append(y_train[index,0])
            
            print("k: {} - j: {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1[0,:]))

            if J_kp1 > J_AGC:
                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu {} and shrinking TR radius to {}".format(mu_kp1[0,:], TR_parameters['radius']))
                J_FOM_list.pop(-1)
                point_rejected = True
                times.append(time.time() - start_time)
            else: 
                print("Improvement good enough: Accpeting the new mu {}".format(mu_kp1[0,:]))

                if TR_parameters['enlarge_radius']:
                    if len(J_FOM_list) > 2:
                        if abs(J_k - J_kp1) > TR_parameters['safety_tolerance']:
                            if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_k - J_kp1)) > TR_parameters['rho']:
                                TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                                print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

                mu_list.append(mu_kp1[0,:])
                J_kernel_list.append(J_kp1)
                mu_k = mu_kp1
                J_k = J_kp1
                times.append(time.time() - start_time)

        if not point_rejected:
            gradient = compute_gradient(kernel_model, mu_k)
            mu_box = mu_k - gradient 
            first_order_criticity = mu_k - projection_onto_range(parameter_space, mu_box)
            normgrad = np.linalg.norm(first_order_criticity)
    
        FOCs.append(normgrad)    
        print("First order critical condition: {}".format(normgrad)) 
        k += 1

    print("\n************************************* \n")

    if k > TR_parameters['max_iterations']:
        print("WARNING: Maximum number of iteration for the TR algorithm reached")
    
    return mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM
    
# #Setting up the  linear problem 
# problem = problems.linear_problem()
# mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
# fom, data = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)
# parameter_space = fom.parameters.space(0, np.pi)
# mu_k = parameter_space.sample_randomly(1)[0]

# mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM = TR_Kernel(fom, TR_parameters={'radius': 0.1, 
#                         'sub_tolerance': 1e-8, 'max_iterations': 7, 'max_iterations_subproblem': 100,
#                         'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
#                         'armijo_alpha': 1e-4, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-10,
#                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True})

# print("Der berechnete Minimierer", mu_list[-1])
# print("Der tatsächliche Wert von mu", J_FOM_list[-1])
# print("Der approxmierte Wert von J", J_kernel_list[-1])
# print("Die benötigte Zeit (online phase) beträgt", times[-1])
# print("FOM eval time", sum(times_FOM))
# print("number of FOM eval", global_counter)

grid, bi = load_gmsh('fin_mesh.msh')

problem = problems.Fin_problem(6)

#setting up optimal parameter 
mu_d = problem.parameter_space.sample_randomly(1, seed=222)[0]
mu_d_as_array = parse_parameter_inverse(mu_d)
mu_d_as_array[0,0] = 0.01
mu_d_as_array[0,1]= 0.1
mu_d = problem.parameters.parse(mu_d_as_array[0,:])

#define gloabal parameter space
#TODO this needs to be done different in later scenarios
parameter_space = problem.parameter_space

print('mu desired:', mu_d)
fom, data, mu_bar = discretizer.discretize_fin_pdeopt_stationary_cg(problem, grid, bi, mu_d, 
                                                            product='fixed_energy',
                                                            add_constant_term=True)

mu_opt = mu_d
mu_opt_as_array = parse_parameter_inverse(mu_opt)
print('Optimal parameter: ', mu_opt_as_array)
J_opt = fom_compute_output(mu_opt)
print('Optimal J: ', J_opt)

print()
mu_start = problem.parameter_space.sample_randomly(1, seed=11)[0]
print('Starting parameter: ', parse_parameter_inverse(mu_start))
J_start = fom_compute_output(mu_start)
print('Starting J: ', J_start)

# for i in range(8):
#     mu_d_as_array[0,5] = i+1
#     mu_d = problem.parameters.parse(mu_d_as_array[0,:])
#     print(mu_d)
#     J = fom_compute_output(mu_d)
#     print("value at ", i, J)

mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM = TR_Kernel(fom, TR_parameters={'radius': 0.1, 
                        'sub_tolerance': 1e-8, 'max_iterations': 20, 'max_iterations_subproblem': 100,
                        'starting_parameter': mu_start, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                        'armijo_alpha': 1e-4, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-10,
                        'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True})
#plt.show(block=True)
print("das sollte rauskommen", mu_opt_as_array)
print("Der berechnete Minimierer", mu_list[-1])
print("Der berechnete Minimieren exakt", J_FOM_list[-1])
print("Der approxmierte Wert von J", J_kernel_list[-1])
print("Die benötigte Zeit (online phase) beträgt", times[-1])
print("FOM eval time", sum(times_FOM))
print("number of FOM eval", global_counter)