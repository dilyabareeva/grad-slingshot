from math import floor, log10
import itertools


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if num == 0:
        return str(0)
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def cdist_mean(U, V, dist):
    sum = 0
    if U.ndim == 3:
        U = U.repeat(V.shape[0], 1, 1, 1)
        UV = list(zip(U, V))
    else:
        UV = list(itertools.product(U, V))
    for u, v in UV:
        sum += dist(
            u.permute((2, 0, 1)).unsqueeze(0), v.permute((2, 0, 1)).unsqueeze(0)
        )

    return sum / len(UV)
