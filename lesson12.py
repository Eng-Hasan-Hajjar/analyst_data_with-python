##lesson 12
import matplotlib.pyplot as plt
import numpy as np


ypoints = np.array([3, 8, 1, 10])

#plt.plot(ypoints, marker = '>')
#plt.plot(ypoints, 'o:r')
##plt.plot(ypoints, 'o-.r')
#plt.plot(ypoints, marker = 'o', ms = 20)
#plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
#plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r',mfc='y')

plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r',mfc='#c9144b')

plt.show()


