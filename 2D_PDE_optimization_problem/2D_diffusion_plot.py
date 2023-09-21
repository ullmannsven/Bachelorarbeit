from pymor.discretizers.builtin.cg import InterpolationOperator
from problems import linear_problem
from pymor.basic import *
import numpy as np

#This creates Figure number 2 of the thesis

problem = linear_problem()
mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
fom, data = discretize_stationary_cg(problem, diameter=1/100, mu_energy_product=mu_bar)
parameter_space = fom.parameters.space(0, np.pi)

mu_init = [0.25, 0.5]
mu_init = problem.parameters.parse(mu_init)

diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(mu_init)
fom.visualize(diff)