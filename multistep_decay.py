import numpy as np
import matplotlib.pyplot as plt


def multi(n0, k, dt=None, duration=None):
    # Default time parameters
    kmax = max(list(k.values()))
    if dt is None:
        dt = np.log(1 / 0.9) / kmax  # such that pmax = 0.1
    if duration is None:
        duration = 100 * dt
    t = np.arange(0, duration, dt)

    # Extract n
    n = list(n0.values())

    # Extract p
    dim = len(n0)
    kmat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            fr = list(n0.keys())[i]
            to = list(n0.keys())[j]
            try:
                kmat[i, j] = k[(fr, to)]
            except Exception:
                pass
    pmat = 1 - (1 / np.exp(dt * kmat))

    # Time-loop
    nuclei = np.zeros((len(t), dim))
    for j in range(len(t)):
        if j == 0:
            nuclei[j] = n
        else:
            # Dice method
            dmat = np.zeros(pmat.shape)
            for i in range(len(n)):
                pmatc = np.expand_dims(np.cumsum(pmat, axis=1)[i], axis=1)
                dice = np.expand_dims(np.random.rand(n[i]), axis=0)

                # Interval checks
                decay = np.sum(pmatc > dice, axis=1)
                decay = np.insert(np.diff(decay), 0, decay[0])
                dmat[i] = decay
            dmat -= np.eye(dim) * np.sum(dmat, axis=1)
            dn = np.sum(dmat, axis=0, dtype=int)
            n += dn
            nuclei[j] = n

    # Make it into a dictionary
    nf = n0
    for i in range(len(n0)):
        keys = list(n0.keys())
        nf[keys[i]] = nuclei.T[i]

    return t, nf


def plotter(t, nf):
    keys = list(nf.keys())
    plt.figure()
    nt = 0
    for i in range(len(nf)):
        plt.plot(t, nf[keys[i]], label=keys[i])
        nt += nf[keys[i]][0]

    # Format
    plt.xlabel('t')
    plt.ylabel('N')
    plt.xlim(0, max(t))
    plt.ylim(0, 1.1 * nt)
    plt.legend()
    plt.show()


# Initial amounts of nuclei
n0 = {
    'A': 1000,
    'B': 3000,
    'C': 500,
    'D': 2000
}

# Decay constants (from, to)
k = {
    ('A', 'B'): 1.0,
    ('A', 'C'): 2.0,
    ('B', 'C'): 1.5,
    ('C', 'D'): 1.2
}

t, nf = multi(n0, k)
plotter(t, nf)
