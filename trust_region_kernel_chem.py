import numpy as np
import math as m
import time 
from itertools import count
import subprocess
global_counter = count()

def parse_ini_file(iter=0):
    """Parses through the relevant lines of the ini.xyz files and returns the current iterate
       as a 2D np.arrary

    Parameters
    ----------
    iter
        The current outer iteration of the TR algorithm.
    
    Return 
        The current iterate mu_k as a 2D numpy array.
    """
    file = open("test_run_1/ini_{}.xyz".format(iter), "r")
    positions = []
    for f in file:
        if any(f.startswith(s) for s in ["N","O","C","H"]):
            f = f[1:]
            coord = f.split()
            for c in coord:
                positions.append(c)
    return np.atleast_2d(positions)

def energy_value():
    file = open("test_run_1/energy", "r")
    for line in file:
        splitted = line.split()
        return splitted[1]
    
def gradient_value(mu_k):
    file = open("test_run_1/gradient", "r")
    skip_counter = 1
    counter = 0
    gradient = np.zeros((1,len(mu_k[0,:])))
    for line in file:
        if skip_counter < 19:
            skip_counter+=1
            pass
        else: 
            splitted = line.split()
            for i in range(len(splitted)):
                gradient[0,3*counter + i] = splitted[i]
                counter+=1
    return gradient

def rename_chm_xtb_file(iter=0):
    """Renames the initial parameter of the current chm_xtb.chm file

    Parameters
    ----------
    iter
        The current outer iteration of the TR algorithm.
    """
    new=iter+1
    with open("test_run_1/opt_xtb_{}.chm".format(iter), "r") as infile, open("test_run_1/opt_xtb_{}.chm".format(new), "w") as outfile:
        for line in infile:
            if "ini_{}".format(iter) in line:
                line = line.replace("ini_{}".format(iter), "ini_{}".format(new))
                outfile.write(line)
            else:
                outfile.write(line)
    infile.close()
    outfile.close()

def write_xyz_file(position, iter=0):
    counter = 0
    new = iter
    with open("test_run_1/ini_0.xyz", "r") as infile, open("test_run_1/ini_{}.xyz".format(new), "w") as outfile:
        for line in infile:
            if any(line.startswith(s) for s in ["N","H","C","O"]):
                outfile.write(line[0])
                line = line[1:]
                splitted = line.split()
                replacements = {}
                for i in range(len(splitted)):
                    replacements[str(splitted[i]).strip()] = str(position[3*counter + i]).strip()
                for key, value in replacements.items():
                    line = line.replace(key,value)
                outfile.write(line)
                counter+=1
            else:
                outfile.write(line)

def Gaussian_kernel_matrix(X,Y,width=2):
    N1 = np.ones((Y.shape[0],1)) @ np.sum(X**2, axis=1).reshape(1,-1)
    N2 = np.sum(Y**2, axis=1).reshape(-1,1) @ np.ones((1,X.shape[0]))
    dist_matrix = N1 - 2 * np.dot(X, Y.T).T + N2
    return np.exp(-width*dist_matrix)

def update_kernel_model(X,Y):
    K = Gaussian_kernel_matrix(X,X)
    alpha = np.linalg.solve(K, Y)
    s_n = lambda x: alpha.T @ Gaussian_kernel_matrix(X,x) 
    return s_n

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

def fom_compute_output(mu, iter=0):
    write_xyz_file(mu, iter)
    rename_chm_xtb_file(iter)
    subprocess.run("cd test_run_1".split())
    subprocess.run("chemsh opt_xtb_{}.chm 1>out 2>err".format(iter).split())
    energy = energy_value()
    gradient = gradient_value()
    return energy, gradient 

def compute_RKHS_norm(mu_k, amount):
    """Approximates the RKHS norm of the FOM that gets to be optimized

    Parameters
    ----------
    mu_k
        The starting parameter of the optimization problem

    amount
        The amount of parameters that should be used to approximate the RKHS norm

    Returns
    -------
    rkhs_norm
        An approximation |rkhs_norm| of the RKHS norm of the FOM
    """
    parameter_dim = len(mu_k[0,:])

    X_train = np.zeros((amount,parameter_dim))
    target_values = np.zeros((amount,1))
    for i in range(amount):
        #TODO quick fix as long as we dont know anything about the parameter space
        max_ = np.max(mu_k)
        min_ = np.min(mu_k)
        mu = np.atleast_2d((max_ - min_) * np.random.random_sample((1,parameter_dim)) + min_)
        X_train[i,:] = mu[0,:]
        
        #Start the ChemShell
        target_values[i,0] = fom_compute_output(mu,iter=(-amount+i))
    
    K = Gaussian_kernel_matrix(X_train, X_train)
    alpha = np.linalg.solve(K, target_values)
    
    return m.sqrt(alpha.T @ K @ alpha)

#TODO make eps as a variable
def compute_gradient(kernel_model, mu_k, X_train, y_train):
    """Approximates the gradient at the parameter |mu_k| using a FD scheme.

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
    # K = kernel_model.kernel.eval(X_train, X_train)
    # alpha = np.linalg.solve(K, y_train)
    # gradient = np.zeros((1, len(X_train[0,:])))
    # for i in range(len(X_train[:,0])):
    #     gradient += alpha[i]*2*kernel_model.kernel_par*(X_train[i,:] - mu_k)*kernel_model.kernel.eval(X_train[i,:], mu_k)

    # return gradient 
    dimension = len(mu_k[0,:])
    gradient = np.zeros((1,dimension))
    eps = 0.01
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
    #TODO check if 3*TR is useful
    idx = []
    for i in range(len(X_train[:,0])):
        if np.linalg.norm(X_train[i,:] - mu_k[0,:]) > 3*TR_parameters["radius"]:
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
    dimension = len(mu_k[0,:])

    if X_train_old is not None and y_train_old is not None:
        X_train, y_train = remove_far_away_points(X_train_old, y_train_old, mu_k, TR_parameters)
        X_train = np.append(X_train, mu_k, axis=0)
        old_len = len(X_train[:,0])
        X_train = projection_onto_range(parameter_space, X_train)
        X_train = remove_similar_points(X_train)
        if old_len == len(X_train[:,0]):
            new_target_value = fom_compute_output(X_train[-1,:])
            y_train = np.append(y_train, np.array(new_target_value, ndmin=2, copy=False), axis=0)
        length_clean = len(X_train[:,0])

        # kernel_model = kernel_model.fit(X_train, y_train, maxIter=length_clean)
        # print("X_train in pounkte", X_train)
        # gradient = compute_gradient(kernel_model, mu_k, X_train, y_train)
        # print("gradd", gradient)
        # return X_train, y_train, gradient
    else: 
        X_train = mu_k
        y_train = np.zeros((1,1))
        y_train[0,0] = fom_compute_output(mu_k)
        length_clean = 1
        # eps = 0.05
        # for j in range(dimension):
        #     unit_vec = np.zeros((1,dimension))
        #     unit_vec[0,j] = 1
        #     fd_point_p = mu_k + eps*unit_vec
        #     fd_point_m = mu_k - eps*unit_vec
        #     X_train = np.append(X_train, fd_point_p, axis=0)
        #     X_train = np.append(X_train, fd_point_m, axis=0)
        # X_train = projection_onto_range(parameter_space, X_train)
        # X_train = remove_similar_points(X_train)
        # length_clean = len(X_train[:,0])
        # y_train = np.zeros((length_clean,1))
        # for i in range(length_clean):
        #     y_train[i,0] = fom_compute_output(X_train[i,:])
        #     next(global_counter)
        # kernel_model = kernel_model.fit(X_train, y_train, maxIter=length_clean)
        
        # gradient = np.zeros((1,dimension))
        # eps = 0.05
        # for j in range(dimension):
        #    unit_vec = np.zeros((1,dimension))
        #    unit_vec[0,j] = 1
        #    gradient[0,j] = (kernel_model.predict(mu_k + eps*unit_vec) - kernel_model.predict(mu_k - eps*unit_vec))/(2*eps)
        # print(X_train)
        # return X_train, y_train, gradient

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
    gradient = compute_gradient(kernel_model, mu_k, X_train, y_train)

    return X_train, y_train, gradient

def create_training_dataset(mu_k, TR_parameters, gradient, X_train_old=None, y_train_old=None, iter=0):
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
    if y_train_old is not None and X_train is not None:
        X_train, y_train = remove_far_away_points(X_train_old, y_train_old, mu_k, TR_parameters)
        length_clean = len(X_train[:,0])
    else: 
        X_train = np.append(X_train, mu_k, axis=0)
        length_clean = 0

    direction = -gradient
    
    new_point = mu_k + 1*direction
    X_train = np.append(X_train, new_point, axis=0)
    new_point = mu_k + 2*direction
    X_train = np.append(X_train, new_point, axis=0)
    new_point = mu_k + 3*direction
    X_train = np.append(X_train, new_point, axis=0)
    
    #X_train = projection_onto_range(parameter_space, X_train)
    X_train = remove_similar_points(X_train)

    for i in range(len(X_train[:,0])-length_clean):
        new_target_value = fom_compute_output(X_train[length_clean+i,:], iter=iter)
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

def optimization_subproblem_BFGS(kernel_model, X_train, y_train, mu_i, TR_parameters, RKHS_norm):
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
    
    gradient = compute_gradient(kernel_model, mu_i, X_train, y_train)
    print("gradient trian set", X_train)
    print(TR_parameters["radius"])
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

        mu_ip1, Jip1, boundary_TR_criterium = armijo_rule(kernel_model, X_train, TR_parameters, mu_i, Ji, direction, RKHS_norm)
        
        if i == 1:
            J_AGC = Jip1
        
        mu_diff = np.linalg.norm(mu_i - mu_ip1) / (np.linalg.norm(mu_i))
        J_diff = abs(Ji - Jip1) / abs(Ji)
        old_mu = mu_i.copy()
        mu_i = mu_ip1
        Ji = Jip1

        old_gradient = gradient.copy()
        gradient = compute_gradient(kernel_model, mu_i, X_train, y_train)
        
        mu_box = mu_i - gradient 
        first_order_criticity = mu_i - projection_onto_range(parameter_space, mu_box)
        normgrad = np.linalg.norm(first_order_criticity)
        
        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)

        i += 1

    print('______ ending BFGS subproblem _______\n')

    return mu_ip1, J_AGC, i, Jip1


def TR_Kernel(TR_parameters=None):
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
        #if 'starting_parameter' not in TR_parameters:
        #    TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
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

    TR_parameters_list = ['radius', 'sub_tolerance', 'max_iterations', 'max_iterations_subproblem',
                         'starting_parameter', 'max_iterations_armijo', 'initial_step_armijo', 
                         'armijo_alpha', 'safety_tolerance', 'FOC_tolerance',
                         'beta_1', 'beta_2', 'rho', 'enlarge_radius']

    for key in TR_parameters.keys():
        assert key in TR_parameters_list

    for (key, val) in TR_parameters.items():
        if key != 'starting_parameter' and key != 'enlarge_radius':
            assert isinstance(val, float) or isinstance(val, int)
        elif key == 'starting_parameter':
            assert isinstance(val, np.ndarray)
        else:
            assert isinstance(val, bool)
            
    mu_k = TR_parameters['starting_parameter']
    
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
    print('\n**************** Starting the offline phase, compute RKHS norm ***********\n')
    RKHS_norm = compute_RKHS_norm(mu_k, amount=30)
    print('\n**************** Done computing the RKHS norm ***********\n')

    start_time = time.time()

    #Compute value and gradient for the first iterate
    tic = time.time()
    J_k, gradient = fom_compute_output(mu_k, iter=0)
    times_FOM.append(time.time()-tic)
    J_FOM_list.append(J_k)

    tic = time.time()
    #X_train, y_train, gradient = create_training_dataset_gradient(mu_k, None, None, TR_parameters, opt_fom_functional, kernel_model)
    X_train, y_train  = create_training_dataset(mu_k, TR_parameters, gradient)
    times_FOM.append(time.time() - tic)

    #Train the kernel model for the new set of centers X_train
    kernel_model = update_kernel_model(X_train, y_train)

    #kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points)

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional {}'.format(J_k))
    print('Initial parameter {}'.format(mu_k))

    k = 1 
    while k <= TR_parameters['max_iterations']:
        print("\n *********** starting iteration number {} ***********".format(k))
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < np.finfo(float).eps:
                print('\n TR-radius below machine precision... stopping')
                break 
        else: 
            if (normgrad < TR_parameters['FOC_tolerance']):
                print('\n Stopping criteria fulfilled... stopping')
                break 

        mu_kp1, J_AGC, j, J_kp1 = optimization_subproblem_BFGS(kernel_model, X_train, y_train, mu_k, TR_parameters, RKHS_norm)
        
        estimator_J = RKHS_norm*power_function(mu_kp1, X_train, kernel)

        if J_kp1 + estimator_J < J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[0,:]))
            
            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            X_train, y_train, gradient = create_training_dataset_gradient(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, kernel_model=kernel_model)
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
            times_FOM.append(time.time()-tic)
            
            #X_train = projection_onto_range(parameter_space, X_train)
            #X_train = remove_similar_points(X_train)
            #num_of_points = len(X_train[:,0])

            print("Updating the kernel model ...\n")
            kernel_model = kernel_model.fit(X_train, y_train, maxIter=num_of_points)

            tic = time.time()
            J_FOM_list.append(fom_compute_output(mu_kp1))
            times_FOM.append(time.time()-tic)
            
            if TR_parameters['enlarge_radius']:
                if len(J_FOM_list) > 2 and abs(J_k - J_kp1) > np.finfo(float).eps:
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
            print("Rejecting the parameter mu {}".format(mu_kp1[0,:]))

            TR_parameters['radius'] *= TR_parameters['beta_1']
            print("Shrinking the TR radius to {}". TR_parameters['radius'])
            point_rejected = True
            times.append(time.time() - start_time)
    
        else: 
            print("Accepting to check if new model is better")

            print("\nSolving FOM for new interpolation points ...")
            tic = time.time()
            X_train, y_train, gradient = create_training_dataset_gradient(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, kernel_model=kernel_model)
            X_train, y_train, num_of_points = create_training_dataset(mu_kp1, X_train, y_train, TR_parameters, opt_fom_functional, gradient)
            times_FOM.append(time.time()-tic)
            
            #X_train = projection_onto_range(parameter_space, X_train)
            #X_train = remove_similar_points(X_train)
            #num_of_points = len(X_train[:,0])

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
                    if len(J_FOM_list) > 2 and abs(J_k - J_kp1) > np.finfo(float).eps:
                            if (k-1 != 0) and ((J_FOM_list[-2] - J_FOM_list[-1])/(J_k - J_kp1)) > TR_parameters['rho']:
                                TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                                print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

                mu_list.append(mu_kp1[0,:])
                J_kernel_list.append(J_kp1)
                mu_k = mu_kp1
                J_k = J_kp1
                times.append(time.time() - start_time)

        if not point_rejected:
            gradient = compute_gradient(kernel_model, mu_k, X_train, y_train)
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
    
mu_k = parse_ini_file(iter=0)
mu_list, J_FOM_list, J_kernel_list, FOCs, times, times_FOM = TR_Kernel(TR_parameters={'radius': 0.1, 
                        'sub_tolerance': 1e-8, 'max_iterations': 8, 'max_iterations_subproblem': 100,
                        'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                        'armijo_alpha': 1e-4, 'safety_tolerance': 1e-16, 'FOC_tolerance': 1e-10,
                        'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.75, 'enlarge_radius': True})

parameter_space = np.zeros(2)
print("Der berechnete Minimierer", mu_list[-1])
print("Der tatsächliche Wert von mu", J_FOM_list[-1])
print("Der approxmierte Wert von J", J_kernel_list[-1])
print("Die benötigte Zeit (online phase) beträgt", times[-1])
print("FOM eval time", sum(times_FOM))
print("number of FOM eval", global_counter)
