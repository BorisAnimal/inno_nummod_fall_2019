import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.sparse import csr_matrix, lil_matrix

boxes = [(23, 37), (23, 88),
         (46, 4), (46, 53),
         (64, 37), (64, 88),
         (87, 4), (87, 53)]

L_houses = [(23, 4), (23, 53),
            (64, 4), (64, 53)]

l_houses = [(46, 17), (46, 68),
            (87, 17), (87, 68)]


def box_house(posx, posy, x, y):
    return posx < x < posx + 6 and posy < y < posy + 6


def l_house(posx, posy, x, y):
    if posx < x < posx + 6 and posy < y < posy + 26:
        return True
    posx = posx - 4
    posy = posy + 20
    return posx < x < posx + 5 and posy < y < posy + 6


def L_house(posx, posy, x, y):
    if posx < x < posx + 6 and posy < y < posy + 26:
        return True
    posx = posx + 4
    return posx < x <= posx + 4 and posy < y < posy + 6


def houses_check(x, y):
    for box in boxes:
        if box_house(box[0], box[1], x, y):
            return True

    for house in L_houses:
        if L_house(house[0], house[1], x, y):
            return True

    for house in l_houses:
        if l_house(house[0], house[1], x, y):
            return True
    return False


def crossSchema(k=0.5,
                h=1,
                lambda1=1.0,
                lambda2=0,
                steps=30000,
                n=100,
                eps=1e-4,
                tau_coef=0.75,
                make_video=False):
    N = n * n
    tau = tau_coef * h ** 2 / (4 * k)
    print("tau: ", tau)

    u = np.zeros((N, 1))
    for x in range(n):
        u[x, 0] = 1

    A = lil_matrix((N, N))

    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    u0coeff = (1 - 4 * (tau * k) / (h ** 2))
    coeffs = [tau * (k / (h ** 2) - lambda1 / (2 * h)),
              tau * (k / (h ** 2) + lambda1 / (2 * h)),
              tau * (k / (h ** 2) - lambda2 / (2 * h)),
              tau * (k / (h ** 2) + lambda2 / (2 * h))]

    print(u0coeff, coeffs)

    for x in range(0, n):
        for y in range(0, n):
            ind = x * n + y
            if x == 0:  # Left bound all the time 1
                A[ind, ind] = 1
                continue

            A[ind, ind] = u0coeff
            for j in range(0, 4):
                x1 = x + dx[j]
                y1 = y + dy[j]
                coeff = coeffs[j]

                if x1 >= n:  # Right bound
                    pass
                elif y1 >= n or y1 == 0:  # Lower and upper bounds
                    A[ind, ind] += coeff
                elif houses_check(x1, y1):  # House bounds
                    A[ind, ind] += coeff
                else:
                    A[ind, x1 * n + y1] = coeff

    A = csr_matrix(A)
    if make_video:
        ims = []
        fig = plt.figure()
        # fig.gca().invert_yaxis()

    for step in range(steps):
        u_new = A * u
        dif = np.max(np.abs(u_new - u))
        u = u_new
        if dif < eps:
            print("Exit on eps! Step: ", step)
            break
        if make_video and step % 25 == 0:
            im = plt.imshow(normalize_img(u), animated=True)
            ims.append([im])
    if make_video:
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                        repeat_delay=1000)
        return u, step, ani
    return u, step


def normalize_img(img, n=100, m=None):
    if m is None:
        m = n
    return np.flip(img.reshape((n, m)).T, axis=0)


if __name__ == '__main__':
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_video', type=int, dest='make_video',
                        help='Record video: 1\nDo not: 0', default=0)
    parser.add_argument('--eps', type=float, dest='eps', default=1e-5)
    parser.add_argument('--tau_coef', type=float, dest='tau_coef', default=0.05)
    args = parser.parse_args()

    if args.make_video:
        u_res, step, vid = crossSchema(make_video=True, eps=args.eps, tau_coef=args.tau_coef)
        vid.save('out.mp4')
    else:
        u_res, step = crossSchema(make_video=False, eps=args.eps, tau_coef=args.tau_coef)

    plt.imshow(normalize_img(u_res))
    plt.title("eps = {} | steps = {} | time = {}".format(args.eps, step, datetime.datetime.now() - start))
    plt.colorbar()
    plt.show()
