import numpy as np
import copy 
import math as m
import time 
import subprocess
import mpmath as mp
from vkoga.kernels import Wendland, Matern, IMQ, Gaussian
from sklearn.metrics import mean_squared_error
import pandas as pd

global_counter = 0

def parse_ini_file(iter, name=None):
    """Parses through the relevant lines of the ini.xyz files and returns the current iterate
       as a 2D np.arrary

    Parameters
    ----------
    iter
        The current outer iteration of the TR algorithm.
    name 
        In case the file has a specific name. 
    
    Returns
    -------
    mu
        The current iterate mu_k as a 2D numpy array.
    """
    if name is not None:
        file = open("{}.xyz".format(name), "r")
    else:
        file = open("optimization_results/ini_{}.xyz".format(iter), "r")
    mu = []
    for f in file:
        if any(f.startswith(s) for s in ["N","O","C","H"]):
            f = f[1:]
            coord = f.split()
            for c in coord:
               mu.append(float(c))
    mu = np.atleast_2d(mu)
    return mu 

def energy_value():
    """ Reads to energy value from the energy file.

    Returns
    -------
    The energy value from the current energy file
    """
    with open("optimization_results/energy", "r") as file:
        for line in file:
            splitted = line.split()
            if len(splitted) > 1:
                return float(splitted[1])
    
def gradient_value(mu_k):
    """Reads the gradient from the current gradient file at parameter |mu_k|

    Parameters
    ----------
    mu_k
        The parameter for which the gradient was evaluated.
    
    Returns
    -------
    The value of the gradient at the current iteration.
    """
    mu_k = np.atleast_2d(mu_k)
    with open("optimization_results/gradient", "r") as file: 
        skip_counter = 1
        counter = 0
        gradient = np.zeros((1,len(mu_k[0,:])))
        for line in file:
            if skip_counter < 19:
                skip_counter+=1
                pass
            elif line.startswith(r"$"):
                pass
            else:
                splitted = line.split()
                for i in range(len(splitted)):
                    gradient[0,3*counter + i] = float(splitted[i].replace('D', 'e'))
                counter+=1
    return np.atleast_2d(gradient)

def rename_chm_xtb_file(iter):
    """Renames the initial parameter of the current chm_xtb.chm file

    Parameters
    ----------
    iter
        The current outer iteration of the TR algorithm.
    """
    with open("optimization_results/opt_xtb_0.chm", "r") as infile, open("optimization_results/opt_xtb_{}.chm".format(iter), "w") as outfile:
        for line in infile:
            if "ini_0.xyz" in line:
                new_line = line.replace("ini_0", "ini_{}".format(iter))
                outfile.write(new_line)
            else:
                outfile.write(line)

def write_xyz_file(mu, iter):
    """ Writes a new .xyz file to store the coordinates of |mu| in the correct format for the chemshell.

    Parameters
    ----------
    mu 
        The current parameter.
    iter
        The current iteration.

    """
    counter = 0
    mu = np.atleast_2d(mu)
    with open("optimization_results/ini_0.xyz", "r") as infile, open("optimization_results/ini_{}.xyz".format(iter), "w") as outfile:
        for line in infile:
            if any(line.startswith(s) for s in ["N","H","C","O"]):
                outfile.write(line[0])
                line = line[1:]
                splitted = line.split()
                replacements = {}
                for i in range(len(splitted)):
                    replacements[str(splitted[i]).strip()] = str(mu[0,3*counter + i]).strip()
                for key, value in replacements.items():
                    line = line.replace(key,value)
                outfile.write(line)
                counter+=1
            else:
                outfile.write(line)

def update_kernel_model(X_train, y_train, kernel):
    """ Builds a new |kernel_model| using the interpolation point set |X_train|. Uses the python library mpmath if the numpy implementation breaks.

    Parameters
    ----------
    X_train
        The interpolation point set at the current iterate.
    y_train 
        The target values of the interpolation problem at the current iterate.
    kernel
       The |kernel| used to interpolate the objective function.

    Returns
    -------
    kernel_model
        The |kernel_model| that interpolates the interpolates function.
    """
    try: 
        K = kernel.eval(X_train, X_train)
        alpha = np.linalg.solve(K, y_train)
        kernel_model = lambda x: np.dot(kernel.eval(x, X_train), alpha)

    except np.linalg.LinAlgError:
        mp.mp.dps = 100
        K = kernel.eval(X_train, X_train)
        alpha = mp.lu_solve(mp.matrix(K.tolist()), mp.matrix(y_train.tolist()))
        kernel_model = lambda x: mp.matrix(kernel.eval(x,X_train)) * alpha
    return kernel_model

def fom_compute_output(mu, iter=0):
    """ This method evaluates the full order model (FOM) at the given parameter |mu|.

    Parameters
    ----------
    mu 
        The parameter for which the FOM is evaluated.
    iter 
        The number of times the FOM got evaluated already.

    Returns 
    -------
    energy
        The energy value at |mu|.
    gradient 
        The gradient value at |mu|.
    """
    if iter != 0:
        write_xyz_file(mu, iter) 
        rename_chm_xtb_file(iter)
    with open('optimization_results/out', 'w') as stdout_file, open('optimization_results/err', 'w') as stderr_file:
        command = ['chemsh', 'opt_xtb_{}.chm'.format(iter)]
        subprocess.run(command, stdout=stdout_file, stderr=stderr_file, cwd='./optimization_results')
    energy = energy_value() + 30
    gradient = gradient_value(mu)
    return energy, gradient 

def compute_RKHS_norm(kernel, mu_k, gradient):
    """Approximates the RKHS norm of the objective function that gets optimized.

    Parameters
    ----------
    kernel
        The reproducing kernel of the RKHS that is used to interpolate the objective function.
    fom 
        The |fom| that gets evaluated. 
    parameter_space
        The |parameter_space| of the full order model which is optimized.

    Returns
    -------
    rkhs_norm
        An approximation of the RKHS norm of the objective function.
    X_train 
        The interpolation point set that approximates the objective function.
    X_train_gradient 
        The gradients of the parameters in |X_train|.
    y_train 
        The target values of the interpolation problem.
    """ 
    amount = 100
    dim = len(mu_k[0,:])
    X_train = np.zeros((amount,dim))
    X_train_gradient = np.zeros((amount, dim))
    target_values = np.zeros((amount,1))

    for i in range(amount):
        mu = np.zeros((1, dim))
        for j in range(dim):
            mu[0,j] = np.random.normal(mu_k[0,j]-gradient[0,j], 0.05)
        X_train[i,:] = mu[0,:]
        target_values[i,0], X_train_gradient[i,:]  = fom_compute_output(mu,iter=(-amount+i))

    K = kernel.eval(X_train, X_train)
    alpha = np.linalg.solve(K, target_values)
    RKHS_norm  =  m.sqrt(np.dot(np.dot(alpha.T,K),alpha))  

    return RKHS_norm, X_train, X_train_gradient, target_values

def compute_gradient(kernel, mu_k, X_train, X_train_gradient, y_train, TR_parameters):
    """Approximates the gradient at the parameter |mu_k| using differentiation of the kernel model.

    Currently implemented for the Gaussian, the IMQ, the quadratic Matern and the second order Wendland kernel.

    Parameters
    ----------
    kernel
        The |kernel| which is used to approximate the objective function.
    mu_k 
        The parameter |mu_k| where the gradient is computed.
    X_train 
        The  interpolation points set at the current iterate.
    X_train_gradient 
        The gradients of the parameters in |X_train|.
    Y_train 
        The target values of the interpolation problem at the current iterate.
    TR_parameters
        The dictionary |TR_parameters| which contains all the parameters of the kernel TR algorithm.
    
    Returns
    -------
    gradient
        An approximation of the |gradient| at the parameter |mu_k|.
    """
    K = kernel.eval(X_train, X_train)
    try:
        alpha = np.linalg.solve(K, y_train)
    except np.linalg.LinAlgError:
        mp.mp.dps = 100
        alpha = mp.lu_solve(mp.matrix(K), mp.matrix(y_train))

    dim = len(mu_k[0,:])
    gradient = np.zeros((1,dim))

    for i in range(len(X_train[:,0])): 
        if np.linalg.norm(X_train[i,:] - mu_k) == 0: 
            return X_train_gradient[i,:] 

    for j in range(dim):
        for i in range(len(X_train[:,0])):
            if "gauss" in kernel.name:
                gradient[0,j] += alpha[i]*2*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j])*np.exp(-(TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k)**2))
            elif "wen" in kernel.name and kernel.name[-1] == str(2): 
                k = 2
                l = int(np.floor(dim/2)) + k + 1
                if 1 - TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k) > 0:
                    gradient[0,j] += alpha[i]*(((1 -TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k))**(l+k))*(2*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j])*(l**2 + 4*l +3) + TR_parameters['kernel_width']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k)*(3*l +6)) + (l+k)*TR_parameters['kernel_width']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k)*(1 - TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k))**(l+k-1))
            elif "imq" in kernel.name: 
                gradient[0,j] += alpha[i]*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j])/((TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k[0,:])**2 + 1)**(1.5))
            elif "mat2" in kernel.name:
                gradient[0,j] += alpha[i]*(3*TR_parameters['kernel_width']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k) + 3*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j]) - 3*TR_parameters['kernel_width']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k) + (TR_parameters['kernel_width']**3)*(X_train[i,j] - mu_k[0,j])*np.linalg.norm(X_train[i,:] - mu_k) - 2*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j]))*np.exp(-TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k))
            elif "mat3" in kernel.name:
                gradient[0,j] += alpha[i]*(15*TR_parameters['kernel_width']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] - mu_k) + 15*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j]) + 15*TR_parameters['kernel_width']*(X_train[i,j] - mu_k[0,j])/np.linalg.norm(X_train[i,:] -mu_k) + 6*(TR_parameters['kernel_width']**3)*(X_train[i,j] - mu_k[0,j])*np.linalg.norm(X_train[i,:] - mu_k) + 12*(TR_parameters['kernel_width']**2)*(X_train[i,j] - mu_k[0,j]) + (TR_parameters['kernel_width']**4)*(X_train[i,j] - mu_k[0,j])*(np.linalg.norm(X_train[i,:] - mu_k)**2) + 3*(TR_parameters['kernel_width']**3)*(X_train[i,j] - mu_k[0,j])*np.linalg.norm(X_train[i,:] - mu_k))*np.exp(-TR_parameters['kernel_width']*np.linalg.norm(X_train[i,:] - mu_k)) 
            else:
                raise NotImplementedError 
    return gradient

def remove_similar_points(X_train, X_train_gradient, y_train, TR_parameters):
    """Removes points from the parameter training set |X_train| if they far away from the current iterate |mu_k|. 

    Parameters
    ----------
    X_train
        The intepolation point set that gets modified.
    X_train 
        The gradient of the parameters in |X_train|.
    y_train
        The target_values of the interpolation problem.
    mu_k
        The current iterate |mu_k|.
    TR_parameters
        The dictionary |TR_parameters| which contains all the parameters of the kernel TR algorithm.

    Returns
    -------
    X_train
        The modified interpolation training set |X_train|.
    X_train_gradient 
        The modified gradient training set |X_train|.
    y_train
        The modified target_values of the interpolation problem.
    """
    idx = []
    num_of_points = len(X_train[:,0])
    for i in range(num_of_points):
        for j in range(i+1,num_of_points):
            if np.linalg.norm(X_train[i,:] - X_train[j,:]) < 0.0001: 
                idx.append(i)

    X_train = np.delete(X_train, (idx), axis=0)
    X_train_gradient = np.delete(X_train_gradient, (idx), axis=0)
    y_train = np.delete(y_train, (idx), axis=0)
            
    return X_train, X_train_gradient, y_train

def remove_far_away_points(X_train, X_train_gradient, y_train, mu_k, num_to_keep):
    """Removes points from the parameter training set |X_train| if they far away from the current iterate |mu_k|. 

    Parameters
    ----------
    X_train
        The intepolation point set that gets modified.
    X_train_gradient 
        The gradients of the parameters in |X_train|.
    y_train
        The target_values of the interpolation problem.
    mu_k
        The current iterate |mu_k|.
    TR_parameters
        The dictionary |TR_parameters| which contains all the parameters of the kernel TR algorithm.

    Returns
    -------
    X_train
        The modified interpolation training set |X_train|.
    y_train
        The modified target_values of the interpolation problem.
    """
    if num_to_keep > len(X_train[:,0]):
        num_to_keep = len(X_train[:,0])

    distances = np.linalg.norm(X_train - mu_k[0,:], axis=1)
    idx_to_keep = np.argsort(distances,)[:num_to_keep]
    idx_to_keep = np.sort(idx_to_keep)

    X_train =  X_train[idx_to_keep,:]
    X_train_gradient = X_train_gradient[idx_to_keep,:]
    y_train =  y_train[idx_to_keep,:]

    return X_train, X_train_gradient, y_train


def create_training_dataset(mu_k, X_train, X_train_gradient, y_train, global_counter, gradient):
    """Adds the points, that are necessary to approximate the gradient, to the training set. 

    Parameters
    ----------
    mu_k
        The current iterate |mu_k|
    X_train
        The training set from the last iteration
    X_train_gradient 
        The gradients of the parameters in |X_train|.
    y_train
        The target values corresponding to the old training set |X_train_old|
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm
    global_counter
        Counter of the amount of FOM
    gradient 
        The |gradient| at the current iterate |mu_k|.

    Returns
    -------
    X_train
        An updated interpolation point se.
    X_train 
        An updated gradient training set.
    y_train
        The target values corresponding to the updated training set |y_train|
    global_counter 
        An increased global counter 
    """
    search_direction = - gradient / np.linalg.norm(gradient)
    
    new_point = mu_k + np.linalg.norm(gradient)*search_direction
    X_train = np.append(X_train, np.atleast_2d(new_point), axis=0)
    new_target_value, new_gradient = fom_compute_output(new_point, iter=global_counter)
    X_train_gradient = np.append(X_train_gradient, np.atleast_2d(new_gradient), axis=0)
    y_train = np.append(y_train, np.atleast_2d(new_target_value), axis=0)
    global_counter += 1

    return X_train, X_train_gradient, y_train, global_counter


def power_function(kernel, mu, X_train):
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
    K = kernel.eval(X_train, X_train)
    kernel_vector = kernel.eval(X_train, mu)

    try:
        lagrange_basis = np.linalg.solve(K, kernel_vector)
    except np.linalg.LinAlgError :
        mp.mp.dps = 100
        lagrange_basis = mp.lu_solve(mp.matrix(K), mp.matrix(kernel_vector))
        
    interpolant = np.dot(lagrange_basis[:,0], kernel_vector[:,0])
    power_val = m.sqrt(abs(kernel.eval(mu,mu)[0,0] - interpolant))
    
    return power_val

def armijo_rule(kernel_model, kernel,  X_train, TR_parameters, mu_i, mu_i_initial, Ji, direction, gradient, RKHS_norm):
    """Computes a new iterate |mu_ip1| s.t it satisfies the armijo conditions.

    Parameters
    ----------
    kernel_model 
        The kernel_model that is used to interpolate the objective function.
    kernel 
        The kernel that is used to interpolate the objective function.
    X_train 
        The interpolation training set at the current iterate.
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm.
    mu_i 
        The current iterate of the BFGS algorithm.
    mu_i_initial
        The first iterate of the BFGS algorithm.
    Ji
        The value of the |kernel_model| at parameter |mu_i|.
    direction
        The descent direction chosen at the current iteration.
    gradient
        The approximation of the gradient at parameter |mu_i|.
    RKHS_norm 
        The approximation of the |RKHS_norm| of the objective function.

    Returns
    -------
    mu_ip1
        The new parameter that satisfies the armijo conditions.
    Jip1
        The value of the |kernel_model| at the new parameter |mu_ip1|.
    boundary_TR_criterium 
        Indicator if the current iterate is closed to the boundary of the Trust-Region.
        
    """
    j = 0
    cos_phi = np.dot(direction, -gradient.T) / (np.linalg.norm(direction)*np.linalg.norm(gradient))
    condition = True
   
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + np.linalg.norm(gradient)*(TR_parameters['initial_step_armijo']**j)*(direction / np.linalg.norm(direction))
        Jip1 = kernel_model(mu_ip1)[0, 0]
        
        power_val = power_function(kernel, mu_ip1, X_train)
        estimator_J = RKHS_norm*power_val
        
        if TR_parameters['advanced']:
            if (Jip1 - Ji) <= (-1)*(TR_parameters['armijo_alpha']*np.linalg.norm(gradient)*np.linalg.norm(mu_ip1 - mu_i)*cos_phi) and abs(estimator_J / Jip1) <= TR_parameters['radius']:   
                condition = False
                print("Armijo and optimization subproblem constraints satisfied at mu: {} after {} armijo iteration".format(mu_ip1[0,:], j))
        else: 
            if (Jip1 - Ji) <= (-1)*(TR_parameters['armijo_alpha']*np.linalg.norm(gradient)*np.linalg.norm(mu_ip1 - mu_i)*cos_phi) and np.linalg.norm(mu_i_initial - mu_ip1) <= TR_parameters['radius']:
                condition = False
                print("Armijo and optimization subproblem constraints satisfied at mu: {} after {} armijo iterations".format(mu_ip1[0,:], j))

        j += 1

    if condition:
        print("Warning: Maximum iteration for Armijo rule reached, proceeding with latest mu: {}".format(mu_i[0,:]))
        mu_ip1 = mu_i
        Jip1 = Ji
        if TR_parameters['advanced']:
            estimator_J = TR_parameters['radius']*Ji
    
    if TR_parameters['advanced']: 
        boundary_TR_criterium = abs(estimator_J/Jip1)
    else:
        boundary_TR_criterium = np.linalg.norm(mu_ip1 - mu_i_initial)

    return mu_ip1, Jip1, boundary_TR_criterium

def compute_new_hessian_approximation(mu_i, mu_old, gradient, gradient_old, B_old):
    """Computes an approximation of the Hessian at parameter |mu_i|.

    Parameters
    ----------
    mu_i 
        The new iterate of the BFGS subproblem 
    mu_old
        The prevoius iterate of the BFGS subproblem
    gradiient 
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
    yk = np.atleast_2d(gradient - gradient_old)
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
        B_new = B_old + ((den + coeff)/(den**2))*skskT  - (HkykskT/den) - (skHkykT/den)
    else: 
        B_new = np.eye(gradient_old.size)

    return B_new

def optimization_subproblem_BFGS(kernel_model, kernel, X_train, X_train_gradient, y_train, mu_i, gradient, TR_parameters, RKHS_norm, iteration):
    """Solves the optimization subproblem of the TR algorithm using a BFGS with constraints.

    Parameters
    ----------
    kernel_model
        The kernel model which is used to approximate the objective function.
    kernel 
        The kernel model which is used to approximate the objective function.
    X_train
        The interpolation training set at the current iteration.
    y_train 
        The target values corresponding to the interpolation training set |X_train|.
    mu_i
        The current iterate of the TR algorithm.
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm.
    RKHS_norm 
        The approximation of the |RKHS_norm| of the objective function.

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
    
    Ji = kernel_model(mu_i)[0,0]
    print("The gradient at point {} is {}".format(mu_i[0,:], gradient[0,:]))

    B = np.eye(mu_i.size)   
    mu_i_initial = copy.deepcopy(mu_i)

    i = 1
    while i <= TR_parameters['max_iterations_subproblem']:
        if i>1:
            if boundary_TR_criterium >= TR_parameters['beta_2']*TR_parameters['radius']:
                print('Boundary condition of TR satisfied, stopping the sub-problem solver now')
                break
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['J_tolerance']:
                print('Subproblem converged: FOC = {}, mu_diff = {}, J_diff = {}'.format(normgrad, mu_diff, J_diff))
                break

        direction = -np.dot(gradient, B.T)
        
        mu_ip1, Jip1, boundary_TR_criterium = armijo_rule(kernel_model, kernel,  X_train, TR_parameters, mu_i, mu_i_initial, Ji, direction, gradient, RKHS_norm)
        
        if i == 1:
            J_AGC = Jip1
        
        mu_diff = np.linalg.norm(mu_i - mu_ip1) / (np.linalg.norm(mu_i))
        J_diff = abs(Ji - Jip1) / abs(Ji)
        old_mu = mu_i.copy()
        mu_i = mu_ip1
        Ji = Jip1
        
        old_gradient = gradient.copy()
        gradient = compute_gradient(kernel, mu_i, X_train, X_train_gradient, y_train, TR_parameters) 
        normgrad = np.linalg.norm(gradient)
        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)
        
        i += 1

    print('______ ending BFGS subproblem _______\n')

    return mu_ip1, J_AGC, i, Jip1, np.atleast_2d(gradient)


def TR_Kernel(kernel, global_counter, TR_parameters=None):
    """The Trust Region kernel algorithm which is to find the minimum of the |opt_fom_functional|.

    Parameters
    ---------- 
    kernel
        The |kernel| used to interpolate the objective function.
    global_counter 
        Counter of the FOM evaluations.
    TR_parameters
        The dictionary |TR_parameters| which contains all the parameters of the TR algorithm.
    
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
        TR_parameters = {'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 20, 'max_iterations_subproblem': 100,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 100, 'initial_step_armijo': 0.75, 
                         'armijo_alpha': 1e-4, 'FOC_tolerance': 1e-8, 'J_tolerance': 1e-16,
                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'kernel_width': 1.5, 'avanced': True}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 20
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 100
        if 'starting_parameter' not in TR_parameters:
            raise ValueError
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 100
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.75
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1e-4
        if 'FOC_tolerance' not in TR_parameters:
            TR_parameters['FOC_tolerance'] = 1e-8
        if 'J_tolerance' not in TR_parameters:
            TR_parameters['J_tolerance'] = 1e-16
        if 'beta_1' not in TR_parameters: 
            TR_parameters['beta_1'] = 0.5
        if 'beta_2' not in TR_parameters:
            TR_parameters['beta_2'] = 0.95
        if 'rho' not in TR_parameters:
            TR_parameters['rho'] = 0.9
        if 'kernel_width' not in TR_parameters: 
            TR_parameters['kernel_width'] = 1.5
        if 'advanced' not in TR_parameters: 
            TR_parameters['advanced'] = True
	    

    TR_parameters_list = ['radius', 'sub_tolerance', 'max_iterations', 'max_iterations_subproblem',
                         'starting_parameter', 'max_iterations_armijo', 'initial_step_armijo', 
                         'armijo_alpha', 'FOC_tolerance', 'J_tolerance',
                         'beta_1', 'beta_2', 'rho', 'kernel_width', 'advanced']

    for key in TR_parameters.keys():
        assert key in TR_parameters_list

    k = 1
    mu_k = TR_parameters['starting_parameter']
    
    J_FOM_list = []
    J_kernel_list = []
    FOCs = []
    times_FOM = []


    mu_list = []
    mu_list.append(mu_k[0,:])

    normgrad = np.inf
    J_diff = np.inf
    point_rejected = False
    success = True
   
    start_time = time.time()
    J_k, gradient_energy = fom_compute_output(mu_k, iter=global_counter)
    times_FOM.append(time.time()-start_time)
    global_counter += 1
    J_FOM_list.append(J_k)

    print('\n*************** Starting the offline phase, compute RKHS norm ************\n') 
    RKHS_norm, X_train, X_train_gradient, y_train = compute_RKHS_norm(kernel, mu_k, gradient_energy)
    print('\n**************** Done computing the RKHS norm ***********\n')

    X_train = np.append(X_train, np.atleast_2d(mu_k), axis=0)
    X_train_gradient = np.append(X_train_gradient, np.atleast_2d(gradient_energy), axis=0)
    y_train = np.append(y_train, np.atleast_2d(J_k), axis=0)

    tic = time.time()
    X_train, X_train_gradient, y_train, global_counter  = create_training_dataset(mu_k, X_train, X_train_gradient, y_train, global_counter, gradient_energy)
    times_FOM.append(time.time() - tic)
    
    K = kernel.eval(X_train, X_train)
    
    try:
        alpha = np.linalg.solve(K, y_train)
        kernel_model = lambda x: np.dot(kernel.eval(x,X_train), alpha)
    except np.linalg.LinAlgError:
        
        mp.mp.dps = 100
        alpha = mp.lu_solve(mp.matrix(K.tolist()), mp.matrix(y_train.tolist()))
        kernel_model = lambda x: np.dot(mp.matrix(kernel.eval(x,X_train)).tolist(), alpha)
        
    J_k = kernel_model(mu_k)[0,0]
    J_kernel_list.append(J_k)

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_k))
    print('Initial parameter {}'.format(mu_k[0,:]))

    while global_counter <= TR_parameters['max_iterations']:
        print("\n *********** starting iteration number {} ***********".format(k))
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < np.finfo(float).eps:
                print('\n TR-radius below machine precision... stopping')
                break 
        else: 
            if success: 
                if normgrad < TR_parameters['FOC_tolerance'] or J_diff < TR_parameters['J_tolerance']:
                    print('\n Stopping criteria fulfilled with FOC= {} and J_diff = {}... stopping'.format(normgrad, J_diff))
                    break 
              
        mu_kp1, J_AGC, j, J_kp1, gradient_energy = optimization_subproblem_BFGS(kernel_model, kernel, X_train, X_train_gradient, y_train, mu_k, gradient_energy, TR_parameters, RKHS_norm, iteration=k-1)

        estimator_J = RKHS_norm*power_function(kernel, mu_kp1, X_train)

        if J_kp1 + estimator_J < J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[0,:]))
    
            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            J_FOM_kp1, gradient_energy_kp1 = fom_compute_output(mu_kp1, iter=global_counter)
            times_FOM.append(time.time()-tic)
            global_counter += 1
            J_FOM_list.append(J_FOM_kp1)

            tic = time.time()
            X_train, X_train_gradient, y_train, global_counter = create_training_dataset(mu_kp1, X_train, X_train_gradient, y_train, global_counter, gradient=gradient_energy_kp1)
            times_FOM.append(time.time()-tic)

            X_train = np.append(X_train, np.atleast_2d(mu_kp1), axis=0)
            X_train_gradient = np.append(X_train_gradient, np.atleast_2d(gradient_energy_kp1), axis=0)
            y_train = np.append(y_train, np.atleast_2d(J_FOM_kp1), axis=0)
            
            X_train, X_train_gradient, y_train  = remove_similar_points(X_train, X_train_gradient, y_train, TR_parameters)
            X_train, X_train_gradient, y_train = remove_far_away_points(X_train, X_train_gradient, y_train, mu_kp1, 100)
            
            print("Updating the kernel model ...\n")
            kernel_model = update_kernel_model(X_train, y_train, kernel)

            if len(J_FOM_list) >= 2 and abs(J_FOM_list[-2] - J_kp1) > np.finfo(float).eps:
                   if ((J_FOM_list[-2] - J_FOM_list[-1])/(J_FOM_list[-2] - J_kp1)) >= TR_parameters['rho']:
                       TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                       print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

            print("k: {} - j: {} - Cost Functional approx: {} - mu: {}".format(k, j, J_kp1, mu_kp1[0,:]))

            mu_list.append(mu_kp1[0,:])     
            J_kernel_list.append(J_kp1)

            J_diff = abs(J_k - J_kp1) / np.max([abs(J_k), abs(J_kp1), 1])
            mu_k = mu_kp1
            J_k = J_kp1
            gradient_energy = gradient_energy_kp1

        elif J_kp1 - estimator_J > J_AGC:
            print("Rejecting the parameter mu {}".format(mu_kp1[0,:]))
            TR_parameters['radius'] *= TR_parameters['beta_1']
            print("Shrinking the TR radius to {}".format(TR_parameters['radius']))
            
            kernel_model = update_kernel_model(X_train, y_train, kernel)
            
            for index, row in enumerate(X_train):
                if np.linalg.norm(row - mu_k[0,:]) < np.finfo(float).eps:
                    gradient_energy = np.atleast_2d(X_train_gradient[index,:])

            J_diff = np.inf
            point_rejected = True
    
        else: 
            print("Building new model to check if proposed iterate mu = {} decreases sufficiently".format(mu_kp1[0,:]))

            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            J_FOM_kp1, gradient_energy_kp1 = fom_compute_output(mu_kp1, iter=global_counter)
            times_FOM.append(time.time()-tic)
            global_counter += 1


            tic = time.time()
            X_train, X_train_gradient, y_train, global_counter = create_training_dataset(mu_kp1, X_train, X_train_gradient, y_train, global_counter, gradient=gradient_energy_kp1)
            times_FOM.append(time.time() - tic)

            X_train = np.append(X_train, np.atleast_2d(mu_kp1), axis=0)      
            X_train_gradient = np.append(X_train_gradient, np.atleast_2d(gradient_energy_kp1), axis=0)
            y_train = np.append(y_train, np.atleast_2d(J_FOM_kp1), axis=0)
            
            X_train, X_train_gradient, y_train = remove_similar_points(X_train, X_train_gradient, y_train, TR_parameters)
            X_train, X_train_gradient, y_train = remove_far_away_points(X_train, X_train_gradient, y_train, mu_kp1, 100)
        
            print("\nUpdating the kernel model ...\n")    
            kernel_model = update_kernel_model(X_train, y_train, kernel)
            
            J_kp1 = kernel_model(mu_kp1)[0, 0]

            if J_kp1 > J_AGC:
               
                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu = {} and shrinking TR radius to {}".format(mu_kp1[0,:], TR_parameters['radius']))
                
                gradient_energy = gradient_energy_kp1

                for index, row in enumerate(X_train):
                    if np.linalg.norm(row -  mu_k[0,:]) < np.finfo(float).eps:
                        gradient_energy = np.atleast_2d(X_train_gradient[index,:])

                J_diff = np.inf
                point_rejected = True
            
            else: 
                print("Improvement good enough: Accpeting the new mu = {}".format(mu_kp1[0,:]))
                
                mu_list.append(mu_kp1[0,:])
                J_FOM_list.append(J_FOM_kp1)
                J_kernel_list.append(J_kp1)

                if len(J_FOM_list) >= 2 and abs(J_FOM_list[-2] - J_kp1) > np.finfo(float).eps:
                        if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_FOM_list[-2] - J_kp1)) >= TR_parameters['rho']:
                            TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                            print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

                J_diff = abs(J_k - J_kp1) / np.max([abs(J_k), abs(J_kp1), 1])
                mu_k = mu_kp1
                J_k = J_kp1
                gradient_energy = gradient_energy_kp1

        if not point_rejected:
            normgrad = np.linalg.norm(gradient_energy)
        else:  
            normgrad = np.inf 
        
        FOCs.append(normgrad)    
        print("First order critical condition: {}".format(normgrad)) 
        k += 1

    print("\n************************************* \n")

    if k > TR_parameters['max_iterations']:
        print("WARNING: Maximum number of iteration for the TR algorithm reached")
    
    return mu_list, np.array(J_FOM_list)-30, np.array(J_kernel_list)-30, FOCs, time.time() - start_time, times_FOM, global_counter


# Default parameters and kernels for the geomtry optimization.

# kernel = Matern(ep=1.5, k=2)

# mu_list, J_FOM_list, J_kernel_list, FOCs, runtime, times_FOM, global_counter = TR_Kernel(kernel, global_counter, TR_parameters={'radius': 0.1, 
#                         'sub_tolerance': 1e-3, 'max_iterations': 20, 'max_iterations_subproblem': 100,
#                         'starting_parameter': mu_0, 'max_iterations_armijo': 100, 'initial_step_armijo': 0.75, 
#                         'armijo_alpha': 1e-4, 'FOC_tolerance': 1e-8, 'J_tolerance': 1e-16,
#                         'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'kernel_width': 1.5, 'advanced': True})


########################################################


def prepare_data(gamma_list):
    """ Creats a dictionary |data| to save relevant information about the optimization algorithm.

    Parameters
    ----------
    gamma_list 
        List of all kernel widths gamma that are used for the optimization. 
    
    Returns 
    -------
    data
        Dictionary |data| to store the results of the optimization algorithm.
    """
    len_gamma = len(gamma_list)
    data = {'times': np.zeros((len_gamma,1)), 'FOC': np.zeros((len_gamma,1)), 'J_error': np.zeros((len_gamma,1)), 
           'counter': np.zeros((len_gamma, 1)), 'mu_error':  np.zeros((len_gamma,1))}
    return data

def optimize_chem(TR_Kernel, kernel_name, gamma_list, TR_parameters, amount_of_iters):
    """ Repeats the optimization |amount_of_iters| times with different starting parameters. 

    Parameters
    ----------
    TR_Kernel 
        The kernel Trust-Region algorithm.
    kernel_name 
        The name of the kernel that is used in the kernel Trust-Region algorithm. 
    gamma_list 
        List of all kernel widths gamma that are used for the optimization.
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm.
    amount_of_iters 
        Amount of times the optimization is done. 

    Returns
    -------
    data
        Dictionary |data| to store results of the optimization algorithm.
    """ 
    mu_k = parse_ini_file(iter=0)
    mu_sol = parse_ini_file(iter=0, name='ini_sol')
    data = prepare_data(gamma_list)
    save_radius = TR_parameters['radius']
    for j in range(len(gamma_list)):
        if kernel_name == 'imq':
            kernel = IMQ(ep=gamma_list[j])
            TR_parameters['kernel_width'] = m.sqrt(gamma_list[j])
        elif kernel_name == 'gauss':
            kernel = Gaussian(ep=gamma_list[j])
            TR_parameters['kernel_width'] = m.sqrt(gamma_list[j])
        elif kernel_name == 'mat2': 
            kernel = Matern(ep=TR_parameters['kernel_width'], k=2) 
            TR_parameters['kernel_width'] = gamma_list[j]
        elif 'wen' in kernel_name and kernel_name[-1] == str(2):
            kernel = Wendland(ep=TR_parameters['kernel_width'], k=2, d=len(mu_k[0,:]))
            TR_parameters['kernel_width'] = gamma_list[j]
        else: 
            raise NotImplementedError

        for i in range(amount_of_iters):
            global_counter = 0
            TR_parameters['starting_parameter'] = mu_k
            TR_parameters['radius'] = save_radius 
            mu_list, J_FOM_list, _ , FOCs, time, _ , global_counter = TR_Kernel(kernel, global_counter, TR_parameters)
            data['mu_error'][j,0] += mean_squared_error(mu_list[-1], mu_sol)
            data['times'][j,0] += time
            data['FOC'][j,0] += FOCs[-1]
            data['counter'][j,0] += global_counter
            data['J_error'][j,0] += abs((J_FOM_list[-1] - 2.39770431)/(2.39770431))

    return data

def report_chem_kernel_TR(data, gamma_list, amount_of_iters):
    """Reports the results of the optimization algorithm. 

    Parameters
    ----------
    data
        Dictionary |data| to store results of the optimization algorithm.
    gamma_list 
        List of all kernel widths gamma that are used for the optimization.
    amount of iters
        Amount of times the optimization is done. 
    """
    data_new = {
        'gamma': np.array(gamma_list).T, 
        'avg. runtime in [s]': data['times'][:,0]/amount_of_iters, 
        'avg. FOM evals.': data['counter'][:,0]/amount_of_iters,
        'avg. MSE in mu': data['mu_error'][:,0]/amount_of_iters,
        'avg. FOC condition': data['FOC'][:,0]/amount_of_iters,
        'avg. error in J': data['J_error'][:,0]/amount_of_iters
    }

    df = pd.DataFrame(data_new)
    print(df)