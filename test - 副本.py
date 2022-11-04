import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import optimize

# y = phidp[0, 35, 13:24].copy()
y = phidp[0, 35].copy()

y1 = y[y>=0]
plt.plot(y1)

# x = np.arange(len(y)).astype(np.float64)
# loc = np.where(y<0)
# x[loc] = np.nan
# plt.plot(x, y)
# plt.show()

# 把【有值的y】单独拿出来平滑，再放回去
# window_length = 5
# y_new= savgol_filter(y[y>=0], window_length = window_length, polyorder = np.int64((window_length-1)/2))
# y[y>=0] = y_new
# plt.plot(x, y)
# plt.show()

# # 平滑y
# window_length = 25
# y_sm = savgol_filter(y, window_length = window_length, polyorder = np.int64((window_length-1)/2))
# plt.plot(x, y_sm)
# # plt.show()

#%%

n = len(y1)
m = 5
sg = np.array(
    [6*(2*i-m-1)/m/(m+1)/(m-1) for i in range(1, m+1)]
    )


I = np.zeros(shape=(n,n))
for i in range(n):
    I[i,i] = 1
A = np.vstack([np.hstack([I, -I]), np.hstack([I, I])])
# add S-G filter
Z = np.zeros(shape=(n-m+1, n))
M = Z.copy()
for i in range(n-m+1):
    M[i, i:i+m] = sg
Aaug = np.vstack([A, np.hstack([Z, M])])
Aub = -Aaug

b = np.hstack([-y1, y1, np.zeros(n-m+1)])
bub = -b

c = np.hstack([np.ones(n), np.zeros(n)])


res = optimize.linprog(c, Aub, bub)
print(res.message)
resx = res.x
y2 = resx[n:]
# plt.plot(y_new)
plt.plot(y2)
#%%
y3 = y2.copy()
y3[2:-2] = 0.1*y2[0:-4] + 0.25*y2[1:-3] + 0.3*y2[2:-2] + 0.25*y2[3:-1] + 0.1*y2[4:]
y3[:2], y3[-2:]= y3[2], y3[-3]
plt.plot(y3)
