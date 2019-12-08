import operator


class AlmostNumpy(list):
    def __check_scalar(self, other):
        return isinstance(other, (int, float))

    def __add__(self, other):
        if self.__check_scalar(other):
            return AlmostNumpy(map(lambda x: x + other, self))
        else:
            return AlmostNumpy(map(operator.add, self, other))

    __radd__ = __add__

    def __sub__(self, other):
        if self.__check_scalar(other):
            return AlmostNumpy(map(lambda x: x - other, self))
        else:
            return AlmostNumpy(map(operator.sub, self, other))

    __rsub__ = __sub__

    def __mul__(self, other):
        if self.__check_scalar(other):
            return AlmostNumpy(map(lambda x: x * other, self))
        else:
            return AlmostNumpy(map(operator.mul, self, other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if self.__check_scalar(other):
            return AlmostNumpy(map(lambda x: x / other, self))
        else:
            return AlmostNumpy(map(operator.truediv, self, other))

    def __rtruediv__(self, other):
        if self.__check_scalar(other):
            return AlmostNumpy(map(lambda x: other / x, self))
        else:
            return AlmostNumpy(map(operator.truediv, other, self))


def solve(equations, reactants):
    sums = [0 for _ in range(len(reactants))]
    for i, equation in enumerate(equations):
        for members in equation:
            prods = 1
            for member in members:
                if isinstance(member, int):
                    prods *= reactants[member]
                else:
                    prods *= member
            sums[i] += prods
    return AlmostNumpy(sums)


# def euler(y0, eq, dt):
#     """
#     Deprecated. Solves single equation
#     """
#     y1 = y0.copy()
#     for i, e in enumerate(eq):
#         y1[i] += dt * solve(e, y0)
#     return y1


def runge_kutta(y0, dt, N, solver):
    for _ in range(N):
        k1 = solver(y0)
        k2 = solver(y0 + dt * k1 / 2)
        k3 = solver(y0 + dt * k2 / 2)
        k4 = solver(y0 + dt * k3)
        y0 = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y0


if __name__ == '__main__':
    # Read input (filth)
    t = float(input())
    n = int(input())
    # Read amount of reactants at t0
    reactants = []
    for _ in range(n):
        reactants.append(float(input()))

    # Read and parse equations
    m = int(input())
    equations = [list() for _ in range(n)]
    for _ in range(m):
        # Split the left and right parts of equation
        left, right = input()[:-1].split('>')
        # Get k
        k, right = right.split(maxsplit=1)
        k = float(k)

        left_els = []
        for member in left.split('+'):
            amount, elem = member.split()
            left_els.append((float(amount), int(elem)))

        delta = [k]
        for am, el in left_els:
            delta.extend([el for _ in range(int(am))])

        for am, elem in left_els:
            cur_delta = delta.copy()
            cur_delta[0] = - cur_delta[0] * am
            equations[elem].append(cur_delta)

        for member in right.split('+'):
            amount, elem = member.split()
            amount, elem = float(amount), int(elem)
            cur_delta = delta.copy()
            cur_delta[0] *= amount
            equations[elem].append(cur_delta)

    reactants = AlmostNumpy(reactants)
    N = min(1150, int(t * 50))  # Grid step
    dt = t / N

    solver = lambda yi: solve(equations, yi)

    reactants = runge_kutta(reactants, dt, N, solver)

    print(" ".join(map(str, reactants)))
