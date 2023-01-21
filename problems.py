from pymor.basic import *
import numpy as np

def linear_problem():
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

    return problem = StationaryProblem(domain, l, diffusion, outputs=[('l2', l * theta_J)])