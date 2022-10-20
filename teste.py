from easyesn import PredictionESN
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("ggplot")

# %%

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

t_y_u = np.genfromtxt('/home/ana/mestrado/projeto/minimos_quadrados/taco-generator-data.csv', delimiter=',')[1:,:]

t = t_y_u[:, 0]
y = t_y_u[:, 1]
u = t_y_u[:, 2]

n = 6

# %%
esn = PredictionESN(n_input=2*n,
                    n_output=1,
                    spectralRadius=1,
                    n_reservoir=150,
                    noiseLevel=0,
                    randomSeed=1,
                    regressionParameters=[1, 1], #[alpha, lambda]
                    solver="rlsqr",
                    bias=1,
                    outputBias=1,
                    feedback=False)

# %%

err = []
WOut_norm = []

# %%

Au = np.flip(u[1:n+1])
Ay = np.flip(y[0:n])
A = np.hstack((Au, Ay)).T # A = [u[n], ..., u[1], y[n-1], ..., y[0]]

for i in range(n, len(t)-2):
    Au = np.roll(Au, 1)
    Au[0] = u[i+1]
    Ay = np.roll(Ay, 1)
    Ay[0] = y[i]
    A = np.hstack((Au, Ay)).T
    x = np.reshape(A, (1,len(A)))
    o = y[i-2]
    o = np.reshape(o, (1,1))
    err.append(esn.fit(x, o, transientTime=0, verbose=0))
    WOut_norm.append(np.linalg.norm(esn._WOut))

PredictionESN.save(esn,'teste')
# %%np.diff
plt.close('all')
ax = plt.figure(1, figsize =(8,4))
#plt.plot(err, label='$e[k]$')
plt.plot(WOut_norm, label='$||W_{out}||$')
#plt.plot(np.diff(WOut_norm, 1), label='$d||W_{out}||$')
plt.xlabel('amostras')
plt.legend()
plt.grid( alpha=0.35)
plt.show()

# %%
o_esn = []
Au = np.flip(u[1:n+1])
Ay = np.flip(y[0:n])
A = np.hstack((Au, Ay)).T # A = [u[n], ..., u[1], y[n-1], ..., y[0]]

for i in range(n, len(t)-2):
    Au = np.roll(Au, 1)
    Au[0] = u[i+1]
    Ay = np.roll(Ay, 1)
    Ay[0] = y[i]
    A = np.hstack((Au, Ay)).T
    x = np.reshape(A, (1,len(A)))
    o_esn.append(float(esn.predict(x)[0]))

# %% plot
o_esn = np.array(o_esn)
plt.close('all')
ax = plt.figure(1, figsize =(8,4))
plt.plot(o_esn, label='$\hat{y}[k]$')
plt.plot(y[n-2:len(o_esn)+n-2], label='${y}[k]$')
plt.xlabel('amostras')
plt.legend()
plt.legend(title='$||err|| = {:.3f}$'.format(np.linalg.norm(o_esn - y[n-2:len(o_esn)+n-2])))
plt.grid(linestyle='--', alpha=0.35)
plt.show()
