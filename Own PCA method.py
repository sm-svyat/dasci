import numpy as np
import matplotlib.pyplot as plt
import random
import math

def disp(distribution):
    '''
    My function of variance calculation.
    Just fun
    '''
    sum = 0
    for item in distribution:
        sum += item**2
    return sum/(len(distribution)-1)

# Create an array of n items
n = 200
x = np.linspace(100, 200, n)

# Create an array of n items with random deviation
y = np.array([i*2 + random.randint(-60, 60) for i in x])
X = np.array([x,y])

# Do centering of arrays
Xcentered = np.array([X[0] - x.mean(), X[1]-y.mean()])

print("Varianse X : ", disp(Xcentered[0]))
print("Varianse Y : ", disp(Xcentered[1]))

# Get standard deviations
stdX = np.std(Xcentered[0])
stdY = np.std(Xcentered[1])
sqrtStd = math.sqrt(stdX**2 + stdY**2)

# Do scaling of arrays
Xcentered[0] = np.array([x/stdX for x in Xcentered[0]])
Xcentered[1] = np.array([y/stdY for y in Xcentered[1]])
meansTuple = (x.mean(), y.mean())

# Plot Centering & Scaling distribution
plt.plot(Xcentered[0], Xcentered[1], 'b.', label='$Centering$ & $Scaling$ $distribution$')
plt.grid()

# Get covariance matrix.
covmat = np.cov(Xcentered)

#diagonal entries - variances, others enteries - covariance
print ("Covariance X and Y: ", np.cov(Xcentered)[0,1])
print('#'*100)

# get the eigenvalues and eigenvalues of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covmat)
print('Eigenvalues of covariance matrix\n', eigenvalues)
print('Eigenvectors of covariance matrix\n', eigenvectors)

# resulting coefficients for transition to new coordinates
a, b, c, d = eigenvectors[0][0], eigenvectors[0][1], eigenvectors[1][0], eigenvectors[1][1]

# newX = a*X + b*Y; newY = c*X + d*Y
newX = np.array([a*x for x in Xcentered[0]]) + np.array([b*y for y in Xcentered[1]])
newY = np.array([c*x for x in Xcentered[0]]) + np.array([d*y for y in Xcentered[1]])

#newX = np.array([a*x*stdX for x in Xcentered[0]]) + np.array([b*y*stdY for y in Xcentered[1]])
#newY = np.array([c*x*stdX for x in Xcentered[0]]) + np.array([d*y*stdY for y in Xcentered[1]])

# create twe vectors for max and min variances
plt.arrow(0, 0, eigenvectors[0][0],  eigenvectors[0][1], head_width=0.3, head_length=0.5, fc='k', ec='k')
plt.arrow(0, 0,  eigenvectors[1][0], eigenvectors[1][1], head_width=0.3, head_length=0.5, fc='k', ec='k')

#plt.arrow(0, 0, 2*math.sqrt(disp(newX))*eigenvectors[0][0],  2*math.sqrt(disp(newX))*eigenvectors[0][1], head_width=0.3, head_length=0.5, fc='k', ec='k')
#plt.arrow(0, 0,  2*math.sqrt(disp(newY))*eigenvectors[1][0],  2*math.sqrt(disp(newY))*eigenvectors[1][1], head_width=0.3, head_length=0.5, fc='k', ec='k')
#plt.xlim([-3*sqrtStd, 3*sqrtStd])
#plt.ylim([-3*sqrtStd, 3*sqrtStd])

print("Variance newX: ", disp(newX))
print("Variance newY: ", disp(newY))

plt.plot(newX, newY, 'r.', label='$New$ $Centering$ & $Scaling$ $distribution$ $rotated$ $along$ $axis$ $with$ $max$ $variance$')
plt.legend()
plt.show()