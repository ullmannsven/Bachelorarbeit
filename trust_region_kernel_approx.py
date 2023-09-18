from pymor.basic import *
import numpy as np
import math as m
import time 
from vkoga.kernels import Gaussian, IMQ
import problems
import discretizer
##import warnings
#from pymor.discretizers.builtin.cg import InterpolationOperator
#from pymor.parameters.base import Mu
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import mean_squared_error
import mpmath as mp

global_counter = 0

#TODO draw function etc in different file, fix for the end of the BA
def draw_convex_hulls(TR_plot_matrix, TR_parameters, iter, X_train, kernel, kernel_model, RKHS_norm):
    xx = np.linspace(0,np.pi,200)
    grid_x, grid_y = np.meshgrid(xx, xx)
    new_array = 'array{}'.format(iter)
    for l in range(200):
        for m in range(200):
            mu_help = np.array([[grid_x[l,m], grid_y[l,m]]])
            power_val_help = power_function(mu_help, X_train, kernel)
            #func_value_help = kernel_model.predict(mu_help)[0, 0]
            func_value_help = kernel_model(mu_help)[0, 0]
            if TR_parameters['radius'] - power_val_help*RKHS_norm/func_value_help >= 0:
                TR_plot_matrix[new_array] = np.vstack((TR_plot_matrix[new_array], mu_help))
    return


def fom_compute_output(mu):
    return fom.output(mu)[0, 0]
    #mu = problem.parameters.parse(mu)
    #return fom.output_functional_hat(mu)

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

def kernel_matrix_mp(X,Y, kernel):
    K = mp.matrix(len(X[:,0]), len(Y[:,0]))
    for i in range(len(X[:,0])):
        for j in range(len(Y[:,0])):
            if "gauss" in kernel.name:
                K[i,j] = mp.exp(-(kernel.ep*mp.norm(X[i,:] - Y[j,:]))**2)
            elif "imq" in kernel.name:
                K[i,j] = 1 / (mp.sqrt(1 + (kernel.ep*mp.norm(X[i,:] - Y[j,:]))**2))
            else: 
                NotImplementedError
    return K


def update_kernel_model(X_train, y_train, kernel, TR_parameters):
    
    try:
        # TR_parameters['width_gauss'] = TR_parameters['width_gauss_init']
        # if "gauss" in kernel.name:
        #     kernel = Gaussian(ep=TR_parameters['width_gauss'])
        # elif "imq" in kernel.name:
        #     kernel = IMQ(ep=TR_parameters['width_gauss'])
        # else:
        #     NotImplementedError
        K = kernel.eval(X_train, X_train)
        alpha = np.linalg.solve(K, y_train)
        kernel_model =  lambda x: kernel.eval(x,X_train) @ alpha  
        
    except np.linalg.LinAlgError:
        # TR_parameters['width_gauss'] = 10*TR_parameters['width_gauss_init']
        # if "gauss" in kernel.name:
        #     kernel = Gaussian(ep=TR_parameters['width_gauss'])
        # elif "imq" in kernel.name:
        #     kernel = IMQ(ep=TR_parameters['width_gauss'])
        # else:
        #     NotImplementedError
        mp.mp.dps = 50
        K = kernel.eval(X_train, X_train)
        alpha = mp.lu_solve(mp.matrix(K.tolist()), y_train)
        kernel_model = lambda x: mp.matrix(kernel.eval(x, X_train).tolist()) @ alpha
        
    return kernel_model

def compute_RKHS_norm(kernel):
    """Approximates the RKHS norm of the FOM that gets to be optimized

    Parameters
    ----------
    kernel
        The reproducing kernel of the RKHS

    Returns
    -------
    rkhs_norm
        An approximation |rkhs_norm| of the RKHS norm of the FOM
    """

    amount = 30

    #TODO dim auch anders berechnebar, ohne diese for loop
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
    
    #TODO try error block einführen?
    K = kernel.eval(X_train, X_train)
    alpha = np.linalg.solve(K, target_values)
    rkhs_norm = m.sqrt(alpha.T @ K @ alpha)

    return rkhs_norm, X_train, target_values

def remove_similar_points(X_train, TR_parameters, iteration):
    """Removes points from the parameter training set |X_train| if they are to close to each other. 

    This method avoids that the resulting kernel matrix of the training set |X_train| is getting singular.

    Parameters
    ----------
    X_train
        The training set which is getting reduced
    TR_parameters
    #TODO
    iter
    #TODO

    Returns
    -------
    X_train
        The cleared training set |X_train|
    """
    idx = []
    num_of_points = len(X_train[:,0])
    for i in range(num_of_points):
        for j in range(i+1,num_of_points):
            if np.linalg.norm(X_train[i,:] - X_train[j,:]) < ((0.75)**iteration)*TR_parameters['eps']:
                idx.append(i)

    X_train = np.delete(X_train, (idx), axis=0)

    return X_train, idx


def compute_gradient(kernel, mu_k, X_train, y_train, TR_parameters, iteration):
    """Approximates the gradient at the parameter |mu_k| using a FD scheme.

    Parameters
    ----------
    kernel
        The |kernel| which is used for approximating the Full Order Model. #TODO
    mu_k 
        The parameter |mu_k| where the gradient is computed.
    X_train 
        The set of interpolation points used to build the current model.
    Y_train 
        The target values of the objective function corresponding to |X_train|.
    TR_parameters
        .. #TODO add
    iter
        #TODO

    Returns
    -------
    gradient
        An approximation of the |gradient| at parameter |mu_k|.
    """
    #TODO check ob entfernen etwas verändert, sollte eig nix 
    X_train, idx = remove_similar_points(X_train, TR_parameters, iteration=iteration)
    y_train = np.delete(y_train, (idx), axis=0)
    X_train, y_train = remove_far_away_points(X_train, y_train, mu_k, 8)
    
    try:
        # TR_parameters['width_gauss'] = TR_parameters['width_gauss_init']
        # if "gauss" in kernel.name:
        #     kernel = Gaussian(ep=TR_parameters['width_gauss'])
        # elif "imq" in kernel.name:
        #     kernel = IMQ(ep=TR_parameters['width_gauss'])
        # else:
        #     NotImplementedError
        K = kernel.eval(X_train, X_train)
        alpha = np.linalg.solve(K, y_train)
    except np.linalg.LinAlgError:
        # TR_parameters['width_gauss'] = 10*TR_parameters['width_gauss_init']
        # if "gauss" in kernel.name:
        #     kernel = Gaussian(ep=TR_parameters['width_gauss'])
        # elif "imq" in kernel.name:
        #     kernel = IMQ(ep=TR_parameters['width_gauss'])
        # else:
        #     NotImplementedError
        mp.mp.dps = 50
        K = kernel.eval(X_train, X_train)
        alpha = mp.lu_solve(mp.matrix(K), y_train)
    
    dim = len(X_train[0,:])
    gradient = np.zeros((1, dim))
    for j in range(dim):
        for i in range(len(X_train[:,0])):
            if "gauss" in kernel.name:
                gradient[0,j] += alpha[i]*2*(TR_parameters['width_gauss']**2)*(X_train[i,j] - mu_k[0,j])*np.exp(-(TR_parameters['width_gauss']*np.linalg.norm(X_train[i,:] - mu_k)**2))
            elif "wen" in kernel.name and kernel.name[-1] == str(2):
                k = 2
                l = int(np.floor(dim/2)) + k + 1
                if 1 - TR_parameters['width_gauss']*np.linalg.norm(X_train[i,:] - mu_k) > 0:
                    gradient[0,j] += alpha[i]*(((1 -TR_parameters['width_gauss']*np.linalg.norm(X_train[i,:] - mu_k))**(l+k))*(2*(TR_parameters['width_gauss']**2)*(X_train[i,j] - mu_k[0,j])*(l**2 + 4*l +3) + TR_parameters['width_gauss']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k)*(3*l +6)) + (l+k)*TR_parameters['width_gauss']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k)*(1 - TR_parameters['width_gauss']*np.linalg.norm(X_train[i,:] - mu_k))**(l+k-1))
            elif "imq" in kernel.name:
                gradient[0,j] += alpha[i]*(TR_parameters['width_gauss']**2)*(X_train[i,j] - mu_k[0,j])/((TR_parameters['width_gauss']*np.linalg.norm(X_train[i,:] - mu_k[0,:])**2 + 1)**(1.5))
            elif "mat2" in kernel.name:
                gradient[0,j] += alpha[i]*(3*TR_parameters['width_gauss']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k) + 3*(TR_parameters['width_gauss']**2)*(X_train[i,j] - mu_k[0,j]) + 3*TR_parameters['width_gauss']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k) + (TR_parameters['width_gauss']**2)*(X_train[i,j] - mu_k[0,j])*np.linalg.norm(X_train[i,:] - mu_k) + 2*(TR_parameters['width_gauss']**2)*(X_train[i,j] - mu_k[0,j]))*np.exp(-TR_parameters['width_gauss']*np.linalg.norm(X_train[i,:] - mu_k))
            else:
                raise NotImplementedError

    return gradient


def remove_far_away_points(X_train, y_train, mu_k, num_to_keep):
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
    if num_to_keep > len(X_train[:,0]):
        num_to_keep = len(X_train[:,0])

    distances = np.linalg.norm(X_train - mu_k[0,:], axis=1)
    idx_to_keep = np.argsort(distances,)[:num_to_keep]
    idx_to_keep = np.sort(idx_to_keep)

    return X_train[idx_to_keep,:], y_train[idx_to_keep, :]


def create_training_dataset(mu_k, X_train, y_train, TR_parameters, global_counter, iter, initial, gradient=None):
    """Adds the points, that are necessary to approximate the gradient, to the training set. 

    Parameters
    ----------
    mu_k
        The current iterate |mu_k|
    X_train
        The training set from the last iteration
    y_train
        The target values corresponding to the old training set |X_train_old|
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    global_counter
    iter

    Returns
    -------
    X_train
        An updated training set
    y_train
        The target values corresponding to the updated training set |y_train|
    gradient
        An approxmiation of the gradient at the parameter |mu_k|
    """
    
    dimension = len(mu_k[0,:])
    num_of_points_old = len(X_train[:,0])
    
    #TODO hochdim verallg.
    for j in range(dimension):
        unit_vec = np.zeros((1,dimension))
        unit_vec[0,j] = 1
        if initial:
            fd_point_p = mu_k + ((0.75)**iter)*TR_parameters['eps']*unit_vec
            fd_point_m = mu_k - ((0.75)**iter)*TR_parameters['eps']*unit_vec
            X_train = np.append(X_train, fd_point_p, axis=0)
            X_train = np.append(X_train, fd_point_m, axis=0)
        else: 
            if gradient[0,j] < 0: 
                fd_point_p = mu_k + ((0.75)**iter)*TR_parameters['eps']*unit_vec
                X_train = np.append(X_train, fd_point_p, axis=0)
            else: 
                fd_point_m = mu_k - ((0.75)**iter)*TR_parameters['eps']*unit_vec
                X_train = np.append(X_train, fd_point_m, axis=0)

    X_train = projection_onto_range(parameter_space, X_train)
    num_of_points = len(X_train[:,0])
    
    for i in range(num_of_points_old, num_of_points):
       new_target_value = fom_compute_output(X_train[i,:])
       y_train = np.append(y_train, np.atleast_2d(new_target_value), axis=0)
       global_counter += 1
    
    return X_train, y_train, global_counter


def power_function(mu, X_train, kernel):
    """Computes the value of the Power Function for the paramter |mu|.

    Parameters
    ----------
    mu
        The parameter |mu| for which the Power function should be evaluated
    X_train
        The training set of the kernel model
    kernel
        The kernel which is used for approximating the objective function J.
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm

    Returns
    -------
    power_val
        The value of the Power Function at parameter |mu|
    """
    try:
        # TR_parameters['width_gauss'] = TR_parameters['width_gauss_init']
        # if "gauss" in kernel.name:
        #     kernel = Gaussian(ep=TR_parameters['width_gauss'])
        # elif "imq" in kernel.name:
        #     kernel = IMQ(ep=TR_parameters['width_gauss'])
        # else:
        #     NotImplementedError
        K = kernel.eval(X_train, X_train)
        kernel_vector = kernel.eval(X_train, mu)
        lagrange_basis = np.linalg.solve(K, kernel_vector)
    except np.linalg.LinAlgError :
        # TR_parameters['width_gauss'] = 10*TR_parameters['width_gauss_init']
        # if "gauss" in kernel.name:
        #     kernel = Gaussian(ep=TR_parameters['width_gauss'])
        # elif "imq" in kernel.name:
        #     kernel = IMQ(ep=TR_parameters['width_gauss'])
        # else:
        #     NotImplementedError
        mp.mp.dps = 50
        K = kernel.eval(X_train, X_train)
        kernel_vector = kernel.eval(X_train, mu)
        lagrange_basis = mp.lu_solve(mp.matrix(K), mp.matrix(kernel_vector))
        
    interpolant = np.dot(lagrange_basis[:,0], kernel_vector[:,0])
    power_val = m.sqrt(abs(kernel.eval(mu,mu) - interpolant))
    #power_val = m.sqrt(kernel.eval(mu,mu) - interpolant)

    return power_val

def armijo_rule(kernel_model, kernel, X_train, TR_parameters, mu_i, mu_i_initial, Ji, direction, gradient, RKHS_norm, iteration):
    """Computes a new iterate |mu_ip1| s.t it satisfies the armijo conditions.

    Parameters
    ----------
    kernel_model 
        The kernel_model that is used to approximate the FOM
    kernel 
        #TODO
    X_train 
        The training set of the kernel model
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    mu_i 
        The current iterate of the BFGS subproblem
    Ji
        The value of the functional which is optimized at parameter |mu_i|
    direction
        The descent direction chosen at the current iteration
    gradient
        The gradient at parameter |mu_i| 
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
    success = True
    j = 0
    cos_phi = np.dot(direction, -gradient.T)[0,0] / (np.linalg.norm(direction)*np.linalg.norm(gradient))
    condition = True
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + np.linalg.norm(gradient)*(TR_parameters['initial_step_armijo']**j)*(direction / np.linalg.norm(direction))
        mu_ip1 = projection_onto_range(parameter_space, mu_ip1)
        
        Jip1 = kernel_model(mu_ip1)[0, 0]
        power_val = power_function(mu_ip1, X_train, kernel)
        estimator_J = RKHS_norm*power_val
        
        if TR_parameters['advanced'] == True:
            if (Jip1 -Ji) <= (-1)*(TR_parameters['armijo_alpha']*np.linalg.norm(gradient)*np.linalg.norm(mu_ip1 - mu_i)*cos_phi) and abs(estimator_J / Jip1) <= TR_parameters['radius']:
                condition = False
                print("Armijo and optimization subproblem constraints satisfied at mu: {} after {} armijo iterations".format(mu_ip1[0,:], j))
        else: 
            if (Jip1 -Ji) <= ((-1)*(TR_parameters['armijo_alpha'] * TR_parameters['initial_step_armijo']**j)*np.linalg.norm(direction)*(np.linalg.norm(mu_ip1 - mu_i))*cos_phi) and np.linalg.norm(mu_ip1 - mu_i_initial) <= TR_parameters['radius']:
                condition = False
                print("Armijo and optimization subproblem constraints satisfied at mu: {} after {} armijo iterations".format(mu_ip1[0,:], j))

        j += 1

    if condition:
        print("Warning: Maximum iteration for Armijo rule reached, proceeding with latest mu: {}".format(mu_i[0,:]))
        success = False
        mu_ip1 = mu_i
        Jip1 = Ji
        if TR_parameters['advanced'] == True:
            estimator_J = TR_parameters['radius']*Ji
    
    if TR_parameters['advanced'] == True:
        boundary_TR_criterium = abs(estimator_J/Jip1)
    else: 
        boundary_TR_criterium = np.linalg.norm(mu_ip1 - mu_i_initial)
    
    return mu_ip1, Jip1, boundary_TR_criterium, success

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

    #Note that this already computes the inverse!
    yk = gradient - gradient_old
    yk = yk[0,:]
    sk = mu_i - mu_old
    sk = sk[0,:]
    den = np.dot(yk, sk)
    
    if den > 0:
        Hkyk = np.dot(B_old,yk)
        coeff = np.dot(yk, Hkyk)
        HkykskT = np.outer(Hkyk, sk)
        skHkykT = np.outer(sk, Hkyk)
        skskT = np.outer(sk, sk)
        B_new = B_old + ((den + coeff)/(den*den) * skskT)  - (HkykskT/den) - (skHkykT/den)
    else: 
        B_new = np.eye(gradient_old.size)

    return B_new

def optimization_subproblem_BFGS(kernel_model, kernel, X_train, y_train, mu_i, TR_parameters, RKHS_norm, iteration):
    """Solves the optimization subproblem of the TR algorithm using a BFGS with constraints.

    Parameters
    ----------
    kernel_model
        The kernel model which is used to approximate the FOM
    kernel 
        #TODO
    X_train
        The training set of the |kernel_model|
    y_train 
        The 
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
    
    Ji = kernel_model(mu_i)[0, 0]
    gradient = compute_gradient(kernel, mu_i, X_train, y_train, TR_parameters, iteration=iteration)
    print("The gradient at point {} is {}".format(mu_i[0,:], gradient[0,:]))
    
    B = np.eye(mu_i.size)
    mu_i_initial = mu_i.copy()
   
    i = 1
    while i <= TR_parameters['max_iterations_subproblem']:
        if i>1:
            if boundary_TR_criterium >= TR_parameters['beta_2']*TR_parameters['radius']:
                print('Boundary condition of TR satisfied, stopping the subproblem solver now and using mu = {} as next iterate'.format(mu_ip1[0,:]))
                break
            elif normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['J_tolerance']:
                print('Subproblem converged at mu = {}, with FOC = {}, mu_diff = {}, J_diff = {}'.format(mu_ip1[0,:], normgrad, mu_diff, J_diff))
                break
            else:
                print('Subproblem not converged (mu = {}, FOC = {}, mu_diff = {}, J_diff = {}), continuing with next armijo line search'.format(mu_ip1[0,:], normgrad, mu_diff, J_diff))
        
        direction = -np.dot(gradient, B.T)
       
        mu_ip1, Jip1, boundary_TR_criterium, success  = armijo_rule(kernel_model, kernel, X_train, TR_parameters, mu_i, mu_i_initial, Ji, direction, gradient, RKHS_norm, iteration)
        
        if i == 1:
            J_AGC = Jip1
        
        #TODO can be done more efficient if armijo reached max iterations, then it will be possible to remove the "if success" in stop of TR algo
        mu_diff = np.linalg.norm(mu_i - mu_ip1) / (np.linalg.norm(mu_i))
        J_diff = abs(Ji - Jip1) / np.max([abs(Jip1,), abs(Ji), 1])
        old_mu = mu_i.copy()
        mu_i = mu_ip1
        Ji = Jip1
        old_gradient = gradient.copy()

        gradient = compute_gradient(kernel, mu_i, X_train, y_train, TR_parameters, iteration=iteration)
        mu_box = mu_i - gradient 
        first_order_criticity = mu_i - projection_onto_range(parameter_space, mu_box)
        normgrad = np.linalg.norm(first_order_criticity)
        
        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)
        
        i += 1

    print('______ ending BFGS subproblem _______\n')

    return mu_ip1, J_AGC, i, Jip1, gradient, success


#TODO remove unnötiges argument?
def TR_Kernel(opt_fom_functional, global_counter, RKHS_norm, TR_parameters=None):
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
        #TODO check these defaults
        TR_parameters = {'radius': 0.5, 'sub_tolerance': 1e-4, 'max_iterations': 30, 'max_iterations_subproblem': 100,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 30, 'initial_step_armijo': 0.5, 
                         'armijo_alpha': 1e-4, 'FOC_tolerance': 1e-10, 'J_tolerance': 1e-10,
                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'eps': 0.075, 'width_gauss': 0.5, 'advanced': True}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.5
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-4
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 30
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 100
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 30
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1e-4
        if 'FOC_tolerance' not in TR_parameters:
            TR_parameters['FOC_tolerance'] = 1e-8
        if 'beta_1' not in TR_parameters: 
            TR_parameters['beta_1'] = 0.5
        if 'beta_2' not in TR_parameters:
            TR_parameters['beta_2'] = 0.95
        if 'rho' not in TR_parameters:
            TR_parameters['rho'] = 0.9
        if 'eps' not in TR_parameters:
            TR_parameters['eps'] = 0.05
        if 'width_gauss' not in TR_parameters:
            TR_parameters['width_gauss'] = 0.75
        if 'advanced' not in TR_parameters: 
            TR_parameters['advanced'] = True

    TR_parameters_list = ['radius', 'sub_tolerance', 'max_iterations', 'max_iterations_subproblem',
                         'starting_parameter', 'max_iterations_armijo', 'initial_step_armijo', 
                         'armijo_alpha', 'FOC_tolerance', 'J_tolerance',
                         'beta_1', 'beta_2', 'rho', 'eps', 'width_gauss', 'advanced']

    for key in TR_parameters.keys():
        assert key in TR_parameters_list

    #for (key, val) in TR_parameters.items():
        # if key != 'starting_parameter' or key != 'advanced':
        #     assert isinstance(val, float) or isinstance(val, int)
        # if key == 'advanced':
        #     assert isinstance(val, bool)
        # else:
        #     #TODO check 
        #     #assert type(val).__name__ == "Mu"
        #     assert isinstance(val, Mu)

    k = 1
        
    mu_k = TR_parameters['starting_parameter']
    mu_k = parse_parameter_inverse(mu_k)
    
    J_FOM_list = []
    J_kernel_list = []
    FOCs = []
    times_FOM = []

    list_delta = {}
    list_delta['0'] = [TR_parameters['radius']]

    mu_list = []
    mu_list.append(mu_k[0,:])

    #TODO only important for the draw part, also outsource this if possible
    TR_plot_matrix = {}
    for i in range(0, TR_parameters['max_iterations']+1):
       new_key = 'array{}'.format(i)
       #TODO make the 2 a variable
       TR_plot_matrix[new_key] = np.zeros((0,2))

    normgrad = np.inf
    J_diff = np.inf
    point_rejected = False
    success = True

    #kernel = Gaussian(ep=TR_parameters['width_gauss'])
    kernel = IMQ(ep=TR_parameters['width_gauss'])
    
    start_time = time.time()
    J_FOM_k = fom_compute_output(mu_k)
    times_FOM.append(time.time()-start_time)
    global_counter += 1
    J_FOM_list.append(J_FOM_k)
    
    X_train = mu_k
    y_train = np.zeros((1,1))
    y_train[0,0] = J_FOM_k
    
    tic = time.time()
    X_train, y_train, global_counter = create_training_dataset(mu_k, X_train, y_train, TR_parameters, global_counter, k-1, True)
    times_FOM.append(time.time() - tic)

    kernel_model = update_kernel_model(X_train, y_train, kernel, TR_parameters)

    J_k = kernel_model(mu_k)[0,0]
    J_kernel_list.append(J_k)

    #draw_convex_hulls(TR_plot_matrix, TR_parameters, 0, X_train, kernel, kernel_model, RKHS_norm)

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_FOM_k))
    print('Initial parameter {}'.format(mu_k[0,:]))

    while k <= TR_parameters['max_iterations']:
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < np.finfo(float).eps:
                print('\n TR-radius below machine precision... stopping')
                break 
        else: 
            if success:
                if normgrad < TR_parameters['FOC_tolerance'] or J_diff < TR_parameters['J_tolerance']:
                    print('\n Stopping criteria fulfilled... stopping')
                    break 

        print("\n *********** starting iteration number {} ***********".format(k))
        
        mu_kp1, J_AGC, j, J_kp1, gradient, success = optimization_subproblem_BFGS(kernel_model, kernel, X_train, y_train, mu_k, TR_parameters, RKHS_norm, iteration=k-1)

        if not success: 
            tic = time.time()
            X_train, y_train, global_counter = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, global_counter, k-1, False, gradient=gradient)
            times_FOM.append(time.time() - tic)
            X_train, idx = remove_similar_points(X_train, TR_parameters, iteration=k-1)
            y_train = np.delete(y_train, (idx), axis=0)
            X_train, y_train = remove_far_away_points(X_train, y_train, mu_kp1, 8)

        estimator_J = RKHS_norm*power_function(mu_kp1, X_train, kernel)
        
        if J_kp1 + estimator_J <= J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[0,:]))
    
            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            J_FOM_kp1 = fom_compute_output(mu_kp1)
            times_FOM.append(time.time()-tic)
            global_counter += 1
            J_FOM_list.append(J_FOM_kp1)

            X_train = np.append(X_train, mu_kp1, axis=0)
            y_train = np.append(y_train, np.atleast_2d(J_FOM_kp1), axis=0)

            X_train, idx = remove_similar_points(X_train, TR_parameters, iteration=k-1)
            y_train = np.delete(y_train, (idx), axis=0)
            X_train, y_train = remove_far_away_points(X_train, y_train, mu_kp1, 8)
            
            print("Updating the kernel model ...\n")
            kernel_model = update_kernel_model(X_train, y_train, kernel, TR_parameters)

            if f"{k-1}" in list_delta:
                    list_delta[f"{k-1}"].append(TR_parameters['radius'])
            else: 
                list_delta[f"{k-1}"] = [TR_parameters['radius']]

            if len(J_FOM_list) >= 2 and abs(J_FOM_list[-2] - J_kp1) > np.finfo(float).eps:
                   if ((J_FOM_list[-2] - J_FOM_list[-1])/(J_FOM_list[-2] - J_kp1)) >= TR_parameters['rho']:
                       TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                       print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

            print("k: {} - j: {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1[0,:]))

            #draw_convex_hulls(TR_plot_matrix, TR_parameters, k, X_train, kernel, kernel_model, RKHS_norm)

            mu_list.append(mu_kp1[0,:])     
            J_kernel_list.append(J_kp1)

            J_diff = abs(J_k - J_kp1) / np.max([abs(J_k), abs(J_kp1), 1])
            mu_k = mu_kp1
            J_k = J_kp1

        elif J_kp1 - estimator_J > J_AGC:
            print("Rejecting the parameter mu {}".format(mu_kp1[0,:]))
            TR_parameters['radius'] *= TR_parameters['beta_1']
            print("Shrinking the TR radius to {}".format(TR_parameters['radius']))
            if f"{k-1}" in list_delta:
                    list_delta[f"{k-1}"].append(TR_parameters['radius'])
            else: 
                list_delta[f"{k-1}"] = [TR_parameters['radius']]

            kernel_model = update_kernel_model(X_train, y_train, kernel, TR_parameters)
            
            J_diff = np.inf
            point_rejected = True
    
        else: 
            print("Building new model to check if proposed iterate mu = {} decreases sufficiently.".format(mu_kp1[0,:]))

            print(X_train)

            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            J_FOM_kp1 = fom_compute_output(mu_kp1)
            times_FOM.append(time.time()-tic)
            global_counter += 1

            X_train = np.append(X_train, mu_kp1, axis=0)
            y_train = np.append(y_train, np.atleast_2d(J_FOM_kp1), axis=0)

            X_train, idx = remove_similar_points(X_train, TR_parameters, iteration=k-1)
            y_train = np.delete(y_train, (idx), axis=0)
            X_train, y_train = remove_far_away_points(X_train, y_train, mu_kp1, 8)
            
            print("\nUpdating the kernel model ...\n")
            kernel_model = update_kernel_model(X_train, y_train, kernel, TR_parameters)
            
            #J_kp1 = J_FOM_kp1 #TODO
            J_kp1 = kernel_model(mu_kp1)[0, 0]

            if J_kp1 > J_AGC:

                if f"{k-1}" in list_delta:
                    list_delta[f"{k-1}"].append(TR_parameters['radius'])
                else: 
                    list_delta[f"{k-1}"] = [TR_parameters['radius']]

                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu = {} and shrinking TR radius to {}".format(mu_kp1[0,:], TR_parameters['radius']))
                
                J_diff = np.inf
                point_rejected = True
            
            else: 
                print("Improvement good enough: Accpeting the new mu = {}".format(mu_kp1[0,:]))
                
                if f"{k-1}" in list_delta:
                    list_delta[f"{k-1}"].append(TR_parameters['radius'])
                else: 
                    list_delta[f"{k-1}"] = [TR_parameters['radius']]

                mu_list.append(mu_kp1[0,:])
                J_FOM_list.append(J_FOM_kp1)
                J_kernel_list.append(J_kp1)

                if len(J_FOM_list) >= 2 and abs(J_FOM_list[-2] - J_kp1) > np.finfo(float).eps:
                        if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_FOM_list[-2] - J_kp1)) >= TR_parameters['rho']:
                            TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                            print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

                #draw_convex_hulls(TR_plot_matrix, TR_parameters, k, X_train, kernel, kernel_model, RKHS_norm)

                J_diff = abs(J_k - J_kp1) / np.max([abs(J_k), abs(J_kp1), 1])
                mu_k = mu_kp1
                J_k = J_kp1

        if not point_rejected:
            gradient = compute_gradient(kernel, mu_k, X_train, y_train, TR_parameters, iteration=k-1)
            mu_box = mu_k - gradient 
            first_order_criticity = mu_k - projection_onto_range(parameter_space, mu_box)
            normgrad = np.linalg.norm(first_order_criticity)
        else: 
            normgrad = np.inf
            
        FOCs.append(normgrad)    
        print("First order critical condition: {}".format(normgrad)) 

        if not point_rejected:
            k += 1

    print("\n************************************* \n")

    if k > TR_parameters['max_iterations']:
        print("WARNING: Maximum number of iteration for the TR algorithm reached")
    
    print("rkhs norm", RKHS_norm)
    
    return mu_list, J_FOM_list, J_kernel_list, FOCs, time.time()-start_time, times_FOM, list_delta, global_counter, TR_plot_matrix, k


#####################################################################################

#Setting up the  linear problem 
problem = problems.linear_problem()
mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
fom, data = discretize_stationary_cg(problem, diameter=1/100, mu_energy_product=mu_bar)
parameter_space = fom.parameters.space(0, np.pi)

amount_of_iterations = 10
gamma_list = [m.sqrt(0.8), m.sqrt(0.9), m.sqrt(1), m.sqrt(1.1), m.sqrt(1.2)]
mu_ref_list = np.load('reference_mu.npy', allow_pickle=True)

result_times = np.zeros((1,len(gamma_list)))
result_times_FOM =np.zeros((1,len(gamma_list)))
result_mu_errors = np.zeros((1,len(gamma_list)))
result_counter = np.zeros((1,len(gamma_list)))
result_FOC = np.zeros((1,len(gamma_list)))
result_J_errors = np.zeros((1,len(gamma_list)))
result_mu = np.zeros((amount_of_iterations*len(gamma_list), 2))
result_iter_used = np.zeros((1,amount_of_iterations))

for j in range(len(gamma_list)):
    counti = 0
    kernel = IMQ(ep=gamma_list[j])
    #kernel = Gaussian(ep=gamma_list[j])
    print('\n**************** Starting the offline phase, compute RKHS norm ***********\n')
    RKHS_norm, _, _ = compute_RKHS_norm(kernel)
    print('\n**************** Done computing the RKHS norm ***********\n')
    for i in range(amount_of_iterations): 
        np.random.seed(i)       
        mu_k = np.random.uniform(0.25, np.pi-0.25, size=(1,2))[0,:]
        mu_k = problem.parameters.parse(mu_k)
        mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM,  list_delta, global_counter, TR_plot_matrix, iter_used = TR_Kernel(fom, global_counter, RKHS_norm, TR_parameters={'radius': 2, 
                        'sub_tolerance': 1e-5, 'max_iterations': 35, 'max_iterations_subproblem': 100,
                        'starting_parameter': mu_k, 'max_iterations_armijo': 40, 'initial_step_armijo': 0.5, 
                        'armijo_alpha': 1e-4, 'FOC_tolerance': 1e-8, 'J_tolerance': 1e-10,
                        'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'eps': 0.03, 'width_gauss': gamma_list[j], 'advanced': True})

        result_times[0,j] += times
        result_times_FOM[0,j] += sum(times_FOM)
        result_FOC[0,j] += FOCs[-1]
        result_counter[0,j] += global_counter
        #result_mu_errors[0,j] += (np.linalg.norm(mu_list[-1] - mu_ref_list[counti,:]) / (np.max([np.linalg.norm(mu_list[-1]), np.linalg.norm(mu_ref_list[counti,:]),1])))
        #result_mu_errors[0,j] = mean_squared_error(mu_list[-1], mu_ref_list[counti,:])
        result_mu_errors[0,j] = np.linalg.norm(mu_list[-1]- mu_ref_list[counti,:])**2 / 2
        #result_mu[counti,:] = mu_list[-1]
        result_J_errors[0,j] += abs((J_FOM_list[-1] - 2.39770431)/(2.39770431))
        #result_iter_used[0,i] = iter_used
        global_counter = 0
        counti += 1

print("av. mu error", result_mu_errors/amount_of_iterations)
print("av. runtime", result_times/amount_of_iterations)
print("av. runtime computing FOM", result_times_FOM/amount_of_iterations)
print("av. FOM evals", result_counter/amount_of_iterations)
print("av FOC", result_FOC/amount_of_iterations)
#print("mu", result_mu)
print("error J", result_J_errors/amount_of_iterations)
#print("av. error J", np.mean(result_J_errors[0,:]))
#print(result_iter_used)


def draw_weird_shapes(TR_plot_matrix, list_mu): 
    fig, ax = plt.subplots()
    #colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(len(mu_list)-1):
        array = 'array{}'.format(i)
        hull = ConvexHull(TR_plot_matrix[array])
        TR_plot_matrix[array] = TR_plot_matrix[array][hull.vertices]
        x = TR_plot_matrix[array][:,0]
        y = TR_plot_matrix[array][:,1]
        ax.plot(list_mu[i][0], list_mu[i][1], 'x', color='red')
        ax.fill(x,y, color='blue', alpha=0.15)
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0,np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mu_1$')
    ax.set_ylabel(r'$\mu_2$')
    plt.show(block=True)

def draw_circles_new(list_delta, list_mu):
    theta = np.linspace(0, 2*np.pi, 500)
    fig, ax = plt.subplots()
    for i in range(len(mu_list)-1):
        circle = plt.Circle((list_mu[i][0], list_mu[i][1]), list_delta[f"{i}"][-1], fill=False, color='blue')
        plt.gca().add_patch(circle)
        x = list_mu[i][0] + list_delta[f"{i}"][-1]*np.cos(theta)
        y = list_mu[i][1] + list_delta[f"{i}"][-1]*np.sin(theta)
        ax.fill(x,y, color='blue', alpha=0.15)
        ax.plot(list_mu[i][0], list_mu[i][1], 'x', color='red')
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0,np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mu_1$')
    ax.set_ylabel(r'$\mu_2$')
    plt.show(block=True)

#draw_circles_new(list_delta, mu_list) 
#draw_weird_shapes(TR_plot_matrix, mu_list)

#TODO eps aus parametern entfernen?? was ist mit den 0.7 ? erwähnen in BA?