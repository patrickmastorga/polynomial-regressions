import numpy as np

"""
taking data such as:
(x1, y1), (x2, y2), ...

the coefficients of the polynomial least square regression curve (line of best fit)
of order N can be derived by solving the (N + 1)x(N + 1) linear system where x, y,
x^2, x^3... represents the sums of their respective values over all data points

**Derived by optimization of multivariable function of N + 1 inputs (represening polynomial coefficients)**

|  1    x   x^2  ...  x^N   |   |  k0  |   |   y    |
|                           |   |      |   |        |
|  x   x^2  x^3  ...  x^N+1 |   |  k1  |   |   yx   |
|  .    .    .         .    | X |  .   | = |  .     |
|  .    .    .         .    |   |  .   |   |  .     |
|  .    .    .         .    |   |  .   |   |  .     |
| x^N  x^N+1     ...  x^2N  |   |  kN  |   |  yx^N  |

"""

def least_squares_polynomial(data, N):
    """ returns array containing coefficients of polynomial least square regression curve """
    # for each power [0, 2N], store the sum of each x coordinate raised to that power
    sums_of_x = [sum(x ** p for x, y in data) for p in range(2 * N + 1)]

    # using the sums of x, assemble the (N + 1)x(N + 1) matrix of coefficients of the linear system
    A = np.empty((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            A[i][j] = sums_of_x[i + j]

    # for each power [0, N], store the sum of each y coordinate times its corresponding x coordinate raised to that power
    sums_of_y = [sum(y * x ** p for x, y in data) for p in range(N + 1)]

    # using the sums of y, assemble the final (N + 1)x1 matrix which equals the polynomial coefficients multiplied by the matrix of coefficients
    B = np.empty((N + 1, 1))
    for i, val in enumerate(sums_of_y):
        B[i][0] = val

    # A x POLYNOMIAL_COEFFICIENTS = B
    # POLYNOMIAL_COEFFICIENTS = A^-1 x B
    pn_coefficients = np.dot(np.linalg.inv(A), B)
    return np.array(pn_coefficients).flatten()

def coefficients_to_string(polynomial_coefficients, round_to=4):
    """ Takes an array of polynomial coefficients and transforms it into an string in the form "y = A + Bx + Cx^2..." """
    # round each value and convert from numpy to python float
    C = [round(val, round_to) for val in polynomial_coefficients]
    y = "y ="

    # add constant term
    if C[0] or len(C) == 1:
        sign = " + " if C[0] >= 0 else " - "
        y += sign + str(abs(C[0]))

    # add first order term
    if C[1]:
        sign = " + " if C[1] >= 0 else " - "
        y += sign + str(abs(C[1])) + "x"

    # add the rest of the terms
    for p, val in enumerate(C[2:], 2):
        if val:
            sign = " + " if val >= 0 else " - "
            y += sign + str(abs(val)) + "x^" + str(p)

    # delete space between first sign and first number and delete first sign if positive
    y = y[:(5 if y[4] == "-" else 4)] + y[6:]
    return y
  
def get_point(polynomial_coefficients, x):
    """ Returns the vakue of the polynomial function at a given x value given an array of polynomial coefficients """
    return sum(val * x ** p for p, val in enumerate(polynomial_coefficients))

if __name__ == "__main__":
    data = [
        [2, 5],
        [-2, 15],
        [4, 7],
        [1, 2],
        [3, 4],
        [5, 6]
    ]
    order = 1

    # INTERESTING: begins to fail when order >= number of points
    # if order = number of points - 1 than function exactly touches every point
    print(coefficients_to_string(least_squares_polynomial(data, order)))