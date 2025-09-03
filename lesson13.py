##lesson13
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

#plt.plot(ypoints, linestyle = 'dotted')
#plt.plot(ypoints, linestyle = 'dashed')
#plt.plot(ypoints, ls = ':')
#plt.plot(ypoints, ls = '--')
#plt.plot(ypoints, ls = '--',color='r')
#plt.plot(ypoints, ls = '--',color="#BAD61B")
#plt.plot(ypoints, linewidth = '20.5')
##plt.show()


"""
y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])

plt.plot(y1)
plt.plot(y2)

plt.show()

"""



"""
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.suptitle("gfdgdfgdf")


plt.title("Sports Watch Data", fontdict = font1, loc = 'left')
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)


plt.show()

