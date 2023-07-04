from pymor.basic import *
import numpy as np
from pymor.basic import ExpressionFunction, LincombFunction, ConstantFunction, StationaryProblem, RectDomain, BitmapFunction
from numbers import Number
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.base import ParameterSpace
from pymor.parameters.base import Parameters

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

    return StationaryProblem(domain, l, diffusion, outputs=[('l2', l * theta_J)])


def Fin_problem(parameter_dimension=2):
    assert parameter_dimension == 2 or parameter_dimension == 6, 'This dimension is not available'
    if parameter_dimension == 2:
        functions = [ExpressionFunction('(2.5 <= x[0]) * (x[0] <= 3.5) * (0 <= x[1]) * (x[1] <=4)* 1.', dim_domain=2),
                     ExpressionFunction('(0 <= x[0]) * (x[0] < 2.5) * (0.75 <= x[1]) * (x[1] <= 1) *1. \
                                        + (3.5 < x[0]) * (x[0] <= 6) * (0.75 <= x[1]) * (x[1] <= 1)* 1. \
                                        + (0 <= x[0]) * (x[0] < 2.5) * (1.75 <= x[1]) * (x[1] <= 2) * 1. \
                                        + (3.5 < x[0]) * (x[0] <= 6) * (1.75 <= x[1]) * (x[1] <= 2) * 1. \
                                        + (0 <= x[0]) * (x[0] < 2.5) * (2.75 <= x[1]) * (x[1] <= 3) *1. \
                                        + (3.5 < x[0]) * (x[0] <= 6) * (2.75 <= x[1]) * (x[1] <= 3) * 1. \
                                        + (0 <= x[0]) * (x[0] < 2.5) * (3.75 <= x[1]) * (x[1] <= 4) *1. \
                                        + (3.5 < x[0]) * (x[0] <= 6) * (3.75 <= x[1]) * (x[1] <= 4) * 1.', dim_domain=2)]
        coefficients = [1,
                        ProjectionParameterFunctional('k', (), ())]
        diffusion = LincombFunction(functions,coefficients)
        parameter_ranges = {'biot': np.array([0.01,1]), 'k': np.array([0.1,10])}
        parameter_type = {'biot': (), 'k': ()}
    elif parameter_dimension == 6:
        functions = [ExpressionFunction('(2.5 <= x[0]) * (x[0] <= 3.5) * (0 <= x[1]) * (x[1] <= 4) * 1.', dim_domain=2),
                     ExpressionFunction('(0 <= x[0]) * (x[0] < 2.5) * (0.75 <= x[1]) * (x[1] <= 1) * \
                                1. + (3.5 < x[0]) * (x[0] <= 6) * (0.75 <= x[1]) * (x[1] <=1) * 1.', dim_domain=2),
                     ExpressionFunction('(0 <= x[0]) * (x[0] < 2.5) * (1.75 <= x[1]) * (x[1] <= 2) * 1. \
                                + (3.5 < x[0]) * (x[0] <= 6) * (1.75 <= x[1]) * (x[1] <= 2) * 1.', dim_domain=2),
                     ExpressionFunction('(0 <= x[0]) * (x[0] < 2.5) * (2.75 <= x[1]) * (x[1] <= 3) *1. \
                                + (3.5 < x[0]) * (x[0] <= 6) * (2.75 <= x[1]) * (x[1] <= 3) * 1.', dim_domain=2),
                     ExpressionFunction('(0 <= x[0]) * (x[0] < 2.5) * (3.75 <= x[1]) * (x[1] <= 4) *1. \
                                + (3.5 < x[0]) * (x[0] <= 6) * (3.75 <= x[1]) * (x[1] <= 4) * 1.', dim_domain=2)]
        coefficients = [ProjectionParameterFunctional('k0'),
                        ProjectionParameterFunctional('k1'),
                        ProjectionParameterFunctional('k2'),
                        ProjectionParameterFunctional('k3'),
                        ProjectionParameterFunctional('k4')]
        diffusion = LincombFunction(functions,coefficients)
        parameter_ranges = {'biot': [0.01,1], 'k0': [0.1,10] , 'k1': [0.1,10],
                            'k2': [0.1,10], 'k3': [0.1,10], 'k4': [0.1,10]}

    domain = RectDomain([[0,0],[6,4]])
    problem = StationaryProblem(
        domain=domain,
        diffusion=diffusion,
        rhs=ConstantFunction(0,2),
        neumann_data=ConstantFunction(-1,2),
        robin_data=(LincombFunction([ConstantFunction(1,2)], [ProjectionParameterFunctional('biot')]), ConstantFunction(0,2)),
        parameter_ranges=parameter_ranges)

    return problem