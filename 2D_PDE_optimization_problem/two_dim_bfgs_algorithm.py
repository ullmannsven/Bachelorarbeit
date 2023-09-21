#from pymor.basic import *
import numpy as np
import time 
from functools import partial
from scipy.optimize import minimize

def fom_objective_functional(fom, mu):
    """ This method evaluates the full order model (FOM) at the given parameter |mu|.

    Parameters
    ----------
    fom
        The FOM that gets evaluated.
    mu 
        The parameter for which the FOM is evaluated.

    Returns 
    -------
    value_FOM
        The value od the FOM at the parameter |mu|.
    """
    value_FOM = fom.output(mu)[0,0]
    return value_FOM

def fom_gradient_of_functional(fom, mu):
    """ This method evaluates the gradient of the full order model (FOM) at the given parameter |mu|.

    Parameters
    ----------
    fom
        The FOM that gets evaluated.
    mu 
        The parameter for which the gradient of the FOM is evaluated.

    Returns 
    -------
    value_FOM_grad
        The value of the gradient of the FOM at the parameter |mu|.
    """
    value_FOM_grad = fom.output_d_mu(fom.parameters.parse(mu), return_array=True, use_adjoint=True)
    return value_FOM_grad

def record_results(function, data, fom, mu=None):
    """ This method evaluates the gradient of the full order model (FOM) at the given parameter |mu|.

    Parameters
    ----------
    function
        The |function| that is evaluated.
    data
        Dictionary |data| to store the results of the optimization algorithm.
    fom 
        The |fom| that is used as an argument of function.
    mu 
        The current iterate |mu| that is used as an argument of function.

    Returns 
    -------
    QoI
        Output of |function|.
    """
    QoI = function(fom, mu)
    data['counter'] += 1
    return QoI

def prepare_data(amount_of_iters):
    """
    Creats a dictionary |data| to save relevant information about the optimization algorithm.

    Parameters
    ----------
    amount_of_iters
        Number of different starting parameters we use.

    Returns
    -------
    data
        Dictionary |data| to store results of the optimization algorithm.
    """
    data = {'times': np.zeros((1,amount_of_iters)), 'J_min': np.zeros((1,amount_of_iters)), 'mu': np.zeros((amount_of_iters,2)), 'counter': 0}
    return data

def optimize_BFGS(J, ranges, amount_of_iters, fom, gradient=None):
    """ Repeats the optimization |amount_of_iters| times with different starting parameters. 

    Parameters
    ----------
    J 
        The objective function that gets optimized. 
    ranges
        The |ranges| of the parameters space. 
    amount_of_iters
        Amount of times the optimization is done. 
    fom 
        The full order model. 
    gradient 
        Gradient information about the full order model

    Returns
    -------
    data
        Dictionary |data| to store results of the optimization algorithm.
    """
    data = prepare_data(amount_of_iters)
    for i in range(amount_of_iters):
        np.random.seed(i)
        mu = np.random.uniform(0.25, np.pi-0.25, size=(1,2))[0,:]
        tic = time.time()
        fom_result = optimize(J, data, ranges, mu, fom, gradient)
        data['times'][0, i] = time.time()-tic
        data['J_min'][0,i] = fom_result.fun
        data['mu'][i,:] = fom_result.x

    return data 

def optimize(J, data, ranges, mu, fom, gradient):
    """ Calls the minimize method from scipy to solve the optimization problem. 

    Parameters
    ----------
    J 
        The objective function that gets optimized. 
    data 
        Dictionary |data| to store results of the optimization algorithm.
    ranges
        The |ranges| of the parameters space. 
    mu 
        The starting parameter |mu|. 
    fom 
        The full order model. 
    gradient 
        Gradient information about the full order model

    Returns 
    -------
    result
        The |result| of one optimization run. 
    """
    if gradient is None:
        jac = None
    else: 
        jac = partial(gradient, fom)

    result = minimize(partial(record_results, J, data, fom),
                      mu,
                      method='L-BFGS-B', jac=jac,
                      bounds=(ranges, ranges),
                      options = {'gtol': 1e-8,'ftol': 1e-10})
    return result

def report(data, amount_of_iters):
    """Reports the results of the optimization algorithm. 

    Parameters
    ----------
    data
        Dictionary |data| to store results of the optimization algorithm.
    amount of iters
        Amount of times the optimization is done. 
    """
    print('\n succeeded!')
    print(f'  mu_min:    {data["mu"][-1,:]}')
    print(f'  J(mu_min): {data["J_min"][0,-1]}')
    print(f'  avg. FOM evals: {data["counter"]/amount_of_iters}')
    print(f'  avg. time:      {sum(data["times"][0,:])/amount_of_iters} seconds')
    print('')