from pymor.basic import *
from pymor.basic import ExpressionFunction, LincombFunction, ConstantFunction, StationaryProblem, RectDomain
from pymor.parameters.functionals import  ExpressionParameterFunctional

def linear_problem():
    ''' This function sets up the 2D optimization problem
    
    This is based on the pyMOR tutorial: Model order reduction for PDE-constrained optimization problems.

    Returns
    -------
    optimization_problem
        The objective functional that gets optimized.
    '''
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

    optimization_problem = StationaryProblem(domain, l, diffusion, outputs=[('l2', l * theta_J)])
    return optimization_problem