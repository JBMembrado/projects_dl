import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(test_x[pred_target.argmax(1) == 1,0], test_x[pred_target.argmax(1) == 1,1])
plt.scatter(test_x[pred_target.argmax(1) == 0,0], test_x[pred_target.argmax(1) == 0,1])
c = plt.Circle((0.5, 0.5), 1/np.sqrt(2*np.pi), fill=False, color='k', linewidth=2)
plt.gcf().gca().add_artist(c)
plt.axis('off')
plt.show()