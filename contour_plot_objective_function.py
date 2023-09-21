from pymor.basic import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def compute_value_matrix(fom, f, x, y):
    """
    Computes the value of the |fom| at the given coordinates |x| and |y|. 

    Parameters
    fom 
        The FOM that gets evaluated.
    f 
        The function that gets called.
    x
        |x| coordinates of the evaluation.
    y
        |y| coordinates of the evaluation.

    Returns 
    -------
    xx 
        meshgrid of x and y.
    yy 
        meshgrid of x and y.
    f_of_x 
        The evaluation of the function |f|.

    """
    f_of_x = np.zeros((len(x), len(y)))
    for ii in range(len(x)):
        for jj in range(len(y)):
            f_of_x[ii][jj] = f(fom, (x[ii], y[jj]))
    xx, yy = np.meshgrid(x, y)
    return xx, yy, f_of_x


def plot_3d_surface(fom, f, x, y, alpha=1):
    """ plots the function f as a 3D plot with contour lines. 

    Parameters 
    ----------
     fom 
        The |fom| corresponding to the function |f|.
    f 
        The function that gets plotted.
    x 
        meshgrid of points to determine the range.
    y 
        meshgrid of points to determine the range.
    alpha 
        fill density of the plot.
    """
    mpl.rcParams['figure.figsize'] = (12.0, 8.0)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['axes.facecolor'] = (0.0, 0.0, 0.0, 0.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, f_of_x = compute_value_matrix(fom, f, x, y)
    ax.contour(x,y,f_of_x, levels=10, zdir='z', offset=1)
    ax.plot_surface(x, y, f_of_x, cmap='Blues',
                    linewidth=0, antialiased=False, alpha=alpha)
    
    ax.view_init(elev=27.7597402597, azim=-39.6370967742)
    ax.set_xlim3d([-0.10457963, 3.2961723])
    ax.set_ylim3d([-0.10457963, 3.29617229])
    ax.set_zlim3d([1,10])
    ax.set_ylabel(r'$\mu_1$')
    ax.set_xlabel(r'$\mu_2$')
    plt.show(block=True)