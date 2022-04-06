import matplotlib.pyplot as plt
from datetime import date
import numpy as np

loss = [[2.4243364334106445, 2.1778557300567627, 2.2128567695617676], [2.18263840675354, 2.126589059829712, 2.196976661682129]]

today = str(date.today())


### loss

loss_ = []
for l in loss:
    for e in l:
        loss_.append(e)
loss = loss_

x = np.arange(1, len(loss)+1, 1)
x *= 2

fig, ax = plt.subplots(1)
ax.set_title(f'Loss {today}')
ax.plot(x, loss) #, color='red', linewidth=3)
ax.set_ylim(0, np.ceil(np.max(loss)))

plt.show()


today = today.replace('-', '')