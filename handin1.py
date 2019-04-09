"""
@author: christopher seay
         sid: s2286181
         email: seay@strw.leidenuniv.nl

@course: numerical recipes in astrophysics
@instructor: van daalen, m.p.

coding structure:
    labels are inside 'problem' function calls. for example, problem_1(case)
    has a comment that specifies its use is for problem 1a.
    procedural programming approach was used here.

TODO: after writing the whole damn thing, don't forget to write to a pdf
"""

import numpy as np
import sys
from astropy.table import Table
import matplotlib
matplotlib.use("Agg") # non-interactive
import matplotlib.pyplot as plt

def random_number_generator(seed):
    """generate random number between [0,1).

    uses XOR 64 bit shift, mwc, and mlcg methods
    to generate a pseudorandom number. multiple
    methods are used to minimize correlation.

    args:
        seed: initial seed

    returns:
        I_j: random float [0,1)
        seed: 'new' seed to continue random number generation

    """

    # random int constants to generate random numbers
    a1 = 21
    a2 = 35
    a3 = 4
    a4 = 182937572
    m = 2**64-1
    m2 = 2**32-1
    a = 9824192
    c = 1223536
    bit64 = 0xffffffffffffffff # 64-bit mask to limit memory use

    # initialize I_j with given seed
    # this seed is newly set after each call
    # start off with a bit of 64-bit XOR-shift
    I_j = seed ^ (seed >> a1) & bit64
    I_j = I_j ^ (I_j << a2)
    I_j = I_j ^ (I_j >> a3)

    # mclg
    I_j = (a*I_j + c) % m

    # mwc
    I_j = a4*(I_j & m2) + (I_j >> 32)

    # xor-shift again
    I_j = I_j ^ (I_j >> a1)
    I_j = I_j ^ (I_j << a2)
    I_j = I_j ^ (I_j >> a3)

    # mwc again
    I_j = a4*(I_j & m2) + (I_j >> 32)

    # finish with mlcg
    # divid by the period to get [0,1)
    I_j = (a*I_j + c) % m
    seed = I_j # set new "seed" for next iteration
    I_j = np.float64(I_j)/m # convert I_j from int to 64-bit float

    return I_j, seed

def calc_factorial(n):
    """return n! (factorial of n).

    returns factorial of a real-valued integer n. for problem 1a.

    TODO: memory overflow error for large int values (e.g, 201).
    gamma function for non-integer real numbers.

    args:
        n: integer

    returns:
        n!

    """

    if n < 0 or type(n) is float:
        raise ValueError('use only real-valued integers for k.')

    # for n usually not within memory limits
    if n > 101:
        result = np.log10(1)
        for i in range(2,n+1):
            result += np.log10(i)

        return np.float64(result)

    # for n within memory limits
    result = 1
    for i in range(2, n + 1):
        result *= i

    return np.float64(result)

def calc_derivative(function,x,step=1e-6):
    """central difference method of differentiation.

    we compute the derivative numerically using the central difference method.
    step = 1e-6 seems to work very well after a bit of testing.
    if 1e-12 error is desirable, 1e-6 is achieved, which is sqrt(err).

    args:
        function: function to be differentiated
        x: maximum x-range; 5 for this assignment
        step: sufficiently small non-zero value to avoid machine error

    returns:
        derivative of function

    """

    fn = function
    h = step
    dx = (fn(x+h)-fn(x-h))/(2*h)
    return dx

def romberg_integral(function,lower_bound,upper_bound,steps):
    """romberg integration calculator.

    using the romberg integration algorithm (which uses the
    trapezoidal, extended trapezoidal, and richardson extrapolation)
    to calculate the integral of a function. creates a table of calculation
    guesses for the integrand, theoretically getting progressively
    closer to the real result. unless the integrand is impressively complex,
    steps param should be between 2-6.

    args:
        function: function to be integrated
        lower_bound: lower-bound of integrand
        upper_bound: upper-bound of integrand
        steps: number of romberg splits

    returns:
        table[-1,-1]: integrated function value

    """

    # shorthand the verbose function input
    fn = function
    a = lower_bound
    b = upper_bound

    table = np.zeros((steps, steps), dtype=np.float64)
    pow_4 = 4 ** np.arange(steps, dtype=np.float64) - 1

    # trapezoidal rule
    h = (b - a)
    table[0, 0] = h * (fn(a) + fn(b)) / 2

    for j in range(1, steps):
        h /= 2

        # extended trapezoidal rule
        table[j, 0] = table[j - 1, 0] / 2
        table[j, 0] += h * np.sum(
            fn(a + i * h) for i in range(1, 2 ** j + 1, 2)
        )

        # richardson extrapolation
        for k in range(1, j + 1):
            table[j, k] = table[j, k - 1] + \
                (table[j, k - 1] - table[j - 1, k - 1]) / pow_4[k]

    return table[-1,-1]

def calc_integral(function,lower_bound,upper_bound,steps):
    """midpoint integrator for improper integrals.

    unfortunately, my romberg integrator doesn't work because the integrand
    doesn't play nice at xmin = 0, so this integrator will do. i'm keeping it
    in the code, however.

    args:
        function: function to be integrated
        lower_bound: lower-bound of integrand
        upper_bound: upper-bound of integrand
        steps: number of romberg splits

    return:
        integrated value

    """

    # shorthand the verbose function input
    fn = function
    a = lower_bound
    b = upper_bound
    h = float(b - a)/steps
    i = 0 # integrand

    for j in range(steps):
        i += fn((a+h/2.0)+j*h)
    i *= h # final area

    return i


def calc_roots(f,a,b,tol,root):
    """find roots of a given function.

    uses bisection method to calculate roots of the given function.
    note: for this assignment, the bounds are actually x = [0,5) but
    bisection does not work if a value is exactly 0, so a << 1 instead.

    args:
        f: function to find root of
        a: lower bound of bracket
        b: upper bound of bracket
        tol: precision tolerance for root (e.g, 1e-8)

    returns:
        c: f(c) = 0
    """

    # brackets
    xl = a
    xr = b
    while np.abs(xl-xr) >= tol:
        c = (xl+xr)/2 # midpoint of bracket
        prod = f(xl)*f(c)
        if prod > root:
            xl = c
        else:
            if prod < root:
                xr = c
    return c

def calc_quantile(sl,quant):
    """calculate quantiles from a sorted list.

    extremely simple algorithm that requires list to be sorted. roughly
    calculates desired quantile

    """
    if quant == 0 or quant > 99:
        raise ValueError('percentiles range from 1-99.')
    q = quant - 1 # counting starts at 0
    quantile = sl[q]
    return np.float64(quantile)

def calc_average_number_of_satallites_per_bin(bin_vals,bins,nhalos):
    """calculate the average number of satellites in a histogram bin.


    Divide bin values by width of bins, then by number of haloes used to create it
    Gives average number of satallies per bin
    """

    avgs = np.zeros(len(bin_vals))

    # need to count and inspect particular indices, so we use enumerate
    # divide each element bin  by the total number of halos to get an avg
    # number of satellites per halo first, then take out of log-log space
    # by dividing by the bin width
    for i, sat in enumerate(bin_vals):
        avg = sat/nhalos
        norm_avg = avg/(bins[i + 1] - bins[i])
        avgs[i] = norm_avg
    return avgs

def poisson_dist(w,n):
    """calculates poisson distribution given by: P_w(k) = [w^k * e^(-w)] / k!

    given a w, k outputs a poisson probability P_w(k). for problem 1a.

    args:
        w: expected value of discrete random variable k.
        k: 0, 1, 2,...

    returns:
        P_w: poisson probability value

    TODO: be able to output (101,200) overflow issue

    """
    k = calc_factorial(n)
    P_w = ((w**n)*(np.exp(-w))/k)
    return np.float64(P_w)

def interp1d_spline(x,xp,fp):
    """1D spline interplator.

    curve fitting using linear polynomials to construct data points within
    specified range of discrete known data points.

    chosen for 2b because my 1D linear interpolator wasn't working.

    TODO: used an online guide for this but it's still not working. using
    a python library instead (numpy.interp(x,xp,fp))
    """

    size = len(xp)

    xpdiff = np.diff(xp)
    fpdiff = np.diff(fp)

    # allocate buffer matrices
    Li = np.zeros(size)
    Li_1 = np.zeros(size-1)
    z = np.zeros(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = np.sqrt(2*xpdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xpdiff[i-1] / Li[i-1]
        Li[i] = np.sqrt(2*(xpdiff[i-1]+xpdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(fpdiff[i]/xpdiff[i] - fpdiff[i-1]/xpdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xpdiff[-1] / Li[i-1]
    Li[i] = np.sqrt(2*xpdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = x.searchsorted(x)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = xp[index],xp[index-1]
    yi1, yi0 = fp[index],fp[index-1]
    zi1, zi0 = z[index],z[index-1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x)**3 + \
         zi1/(6*hi1)*(x-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x)
    return f0

# def linear_interp_3D():
#     """linearly interpolate for a given 3D function.
#
#     """
#
#     continue

# quicksort
def partition(xs, start, end):
    follower = leader = start
    while leader < end:
        if xs[leader] <= xs[end]:
            xs[follower], xs[leader] = xs[leader], xs[follower]
            follower += 1
        leader += 1
    xs[follower], xs[end] = xs[end], xs[follower]
    return follower

def _quicksort(xs, start, end):
    if start >= end:
        return
    p = partition(xs, start, end)
    _quicksort(xs, start, p-1)
    _quicksort(xs, p+1, end)

def quicksort(xs):
    _quicksort(xs, 0, len(xs)-1)
# end quicksort

def random_sampling(function,xmin,xmax,ymin,ymax,sample_size,seed):
    """random sample distribution using rejection sampling.


    creates a random sampling distribution from desired function
    with set size. utilizes rejection sampling to find valid points.
    for this assignment, sample_x will serve as radii, with sample_y the
    output of n(r). this is essentially junk data, but this sampling
    algorithm may be used in the future.

    TODO: had a slice sampling procedure but couldn't figure out
    the inverse function. not sure if there is one. i couldn't tell
    from the horizontal line test using w-alpha

    args:
        function: function to be sampled from
        xmin: minimum in range of possible x-values
        xmax: maximum in range of possible x-values
        ymin: maximum in range of possible y-values
        ymax: maximum in range of possible y-values
        sample_size: desired number of sample points in distribution

    returns:
        sample_x:
        sample_y:

    """
    # fn = function
    # fn_inv = inv_function # actual inverse goes here
    # samples = np.zeros(iter)
    # x = 0
    # for i in range(iter):
    #    u = rand(0, fn(x))
    #    x_lo, x_hi = fn_inv(u)
    #    x = rand(x_lo, x_hi)
    #    samples[i] = x

    # return samples

    rand = random_number_generator # function alias

    fn = function
    sample_x = np.zeros(sample_size)
    sample_y = np.zeros(sample_size)
    samples = 0 # initializes size of accepted sample points
    while samples < sample_size:
        x,seed = rand(seed)
        y,seed = rand(seed) # get next random number
        sx = (xmax-xmin)*x+xmin
        sy = (ymax-ymin)*y+ymin

        if sy <= fn(sx):
            sample_x[samples] = sx
            sample_y[samples] = sy
            samples += 1

    return sample_x, sample_y

def generate_satellite_profile(f,nsats,seed):
    """rejection sampler to generate 3D positions (r,theta,phi) of satellites.

    generates a 3D profile for satellites using a rejection sampler for the
    radial component and a random number generator for the polar,
    azimuthal components (theta, phi respectively).

    args:
        f: function to be used
        nsats: number of satellite galaxies
        seed: random seed

    returns:
        r,t,p 3D position for satellite

    """

    rand = random_number_generator # function alias

    # set x,y range for sampling
    xmin = 0
    ymin = 0
    xmax = 5
    ymax = 5

    # nsats different satellites
    # x_sats,y_sats = random_sampling(f,xmin,xmax,ymin,ymax,nsats,seed)
    r_sats = random_sampling(f,xmin,xmax,ymin,ymax,nsats,seed)[0]
    theta_sats = np.zeros(nsats)
    phi_sats = np.zeros(nsats)
    for i in range(nsats):
        t,seed = rand(seed)
        p,seed = rand(seed) # get next random number
        # multiply by pi, 2pi for polar, azimuthal angle
        theta_sats[i] = np.pi*t
        phi_sats[i] = 2*np.pi*p

    return r_sats, theta_sats, phi_sats

def generate_halo_profile(nsats,nhalos,f,seed):
    """generate halo containing satellite galaxies with 3D positions.

    halo contains nsats number of satellites

    """

    halos = []
    r = []
    t = []
    p = []
    # np.shape(r) = (nhalos,nsats)
    for i in range(nhalos):
        ra,th,ph = generate_satellite_profile(f,nsats,seed)
        halos.append([ra,th,ph])
        r.append(ra)

    # we want all satellites in one big list
    r = np.concatenate(r)

    return halos,r

def rand_thousand_plot(x,x_1):
    fig = plt.figure(figsize=(7,5))
    plt.scatter(x,x_1)
    plt.xlabel('$x_i$',fontsize=14)
    plt.ylabel('$x_{i+1}$',fontsize=14)
    plt.savefig('plt1.png',format='png')

def rand_million_plot(x_mil):
    fig = plt.figure(figsize=(7,5))
    b = np.linspace(0.0,1.0,20)
    plt.title('1 million random numbers distribution')
    plt.hist(x_mil,color='k',bins=b,histtype='step')
    plt.savefig('plt2.png',format='png')

def interp1d_plot(x_range,func_vals,interp):
    fig = plt.figure(figsize=(7,5))
    x = x_range
    y = func_vals
    plt.xlabel('$\log$(x)',fontsize=14)
    plt.ylabel('$\log$(n)',fontsize=14)
    plt.xscale('log') # log scale for x
    plt.yscale('log') # log scale for y
    plt.scatter(x,y,alpha=0.5,c='b',label='data')
    plt.plot(x,interp,c='k',label='interpolated values')
    plt.legend(frameon=False,loc='best')
    plt.savefig('plt3.png',format='png')

def halos_plot(f,r,nhalos):
    fig = plt.figure(figsize=(7,5))
    xmin = 1e-4
    xmax = 5
    log_xmin = np.log10(xmin)
    log_xmax = np.log10(xmax)
    b = np.logspace(xmin,xmax,21) # 1e-4, x_max = 5
    data = np.logspace(xmin,xmax,20)
    N_x_dist = np.arange(1.e-4,5,0.001) # 1000 vals
    N_dist = f(N_x_dist)
    bin_vals,bins,patches = plt.hist(r,bins=b,histtype='step')
    plt.cla()
    new_bin_vals = calc_average_number_of_satallites_per_bin(bin_vals,
                                                            bins,
                                                            nhalos)
    plt.title('$\log$-$\log$ of N(x)')
    plt.xlabel('$\log$[x]',fontsize=14)
    plt.ylabel('$\log$[N(x)]',fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(N_x_dist,N_dist,c='b')
    plt.hist(data,weights=new_bin_vals,
                bins=b,color='k',histtype='step')
    plt.savefig('plt4.png',format='png')
# def write_to_file():
#     continue

def problem_1(seed):

    rand = random_number_generator # function alias
    thou = 1000
    mil = 1000000 # np.zeros won't accept 1e6

    # 1a
    poisson_out_1 = poisson_dist(1,0)
    poisson_out_2 = poisson_dist(5,10)
    poisson_out_3 = poisson_dist(3,21)
    poisson_out_4 = poisson_dist(2.6,40)
    print('poisson results: {}, {}, {}, {}\n'.format(poisson_out_1,
                                                    poisson_out_2,
                                                    poisson_out_3,
                                                    poisson_out_4))

    # 1b
    x = np.zeros(thou)
    x_1 = np.zeros(thou)
    for i in range(len(x)):
        x[i],seed = rand(seed)
        x_1[i],seed = rand(seed)
    rand_thousand_plot(x,x_1)
    x_mil = np.zeros(mil)
    for i in range(len(x_mil)):
        x_mil[i],seed = rand(seed)
    rand_million_plot(x_mil)
    print('rng plots made.\n')


def problem_2(seed):

    rand = random_number_generator # function alias

    # a,b,c are greater than [0,1] from random number generator, so we
    # multiply by (ub-lb)*rand + lb to have a random number in the given range
    # notice: ub/lb --> upper/lower bound
    nhalos = 1000 # number of halos each containing 100 satellites
    nsats = 100 # number of satellites for all of problem 2
    x,seed = rand(seed)
    y,seed = rand(seed)
    z,seed = rand(seed) # x,y,z update the rng for a,b,c constants
    a = 1.4 * x + 1.1
    b = 1.5 * y + 0.5
    c = 2.5 * z + 1.5 # a, b, c used for all of problem 2

    # 2a
    # we can reduce the dimensions of the integrand from 3D -> 1D using
    # phi, theta symmetry
    n_r = lambda x: 4*np.pi*x**2*(x/b)**(a-3)*np.e**(-(x/b)**c)
    lower_bound = 0
    upper_bound = 5
    steps = 5
    int_f = calc_integral(n_r,lower_bound,upper_bound,steps)
    A = 1/int_f
    print('constants: A, a, b, c : {}, {}, {}, {}\n'.format(A,a,b,c))

    # 2b
    # list comprehension is faster than for loop and working with tuples is
    # also slightly faster than with lists
    x_range = (1e-4,1e-2,1e-1,1,5)
    xi = np.log10(x_range)
    n_r_vals = [n_r(x) for x in x_range]
    interp = np.interp(xi,x_range,n_r_vals)
    # interp = interp1d_spline(xi,x_range,func_vals)
    interp1d_plot(x_range,n_r_vals,interp)
    print('1D interpolation plotted.\n')

    # 2c
    n_x = lambda x: A*nsats*(x/b)**(a-3)*np.e**(-(x/b)**c)
    dx = calc_derivative(n_x,b)
    analytic_dx = -A*nsats*(c-a+3)/(b*np.e)
    # by hand and confirmed using W-alpha
    print('numerical dn(x)/dx: {}, analytical dn(x)/dx: {}\n'\
            .format(np.round(dx,12),np.round(analytic_dx,12)))

    # 2d
    halo = generate_satellite_profile(n_x,nsats,seed)
    print('single halo profile in (r,theta,phi) coords.:\n{}'\
            .format(halo))

    # 2e
    N_x = lambda x: n_x(x)*4*np.pi*x**2
    halos,r = generate_halo_profile(nsats,nhalos,N_x,seed)
    halos_plot(N_x,r,nhalos)

    # 2f
    a = 1e-8 # something really close to 0 but not 0
    b = 5
    root = 0.5
    tolerance = 1e-6
    root = calc_roots(N_x,a,b,tolerance,root)
    print('root: {}\n'.format(root))

    # 2g
    # quantiles desired
    p16 = 16
    p50 = 50
    p84 = 84
    r_sorted = halos[0][0] # radial bin of first halo, want it for largest number of galaxies
    quicksort(r_sorted)
    q1 = calc_quantile(r_sorted,p16)
    q2 = calc_quantile(r_sorted,p50)
    q3 = calc_quantile(r_sorted,p84)
    print('16, 50, 84 quantiles: {}, {}, {}'.format(q1,q2,q3))

    # 2h normalization and 3d interp
    print('2h not implemented.')

def problem_3(seed):
    print('this shit hard fam. problem 3 not implemented.')

def main():
    seed = 0 # random seed for assignment

    print('nur assignment 1. beginning, please wait......................\n')

    print('seed is {}\n'.format(seed))
    # q1 outputs
    problem_1(seed)

    # q2 outputs
    problem_2(seed)

    # q3 outputs
    problem_3(seed)

    print('completed. please take a look at the pdf.\n')

if __name__ == "__main__":
    sys.exit(main())
