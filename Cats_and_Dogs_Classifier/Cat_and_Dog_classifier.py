import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset: [whisker_length, ear_flappiness]
# Cats: long whiskers, less flappy ears
cats = np.array([
    [6, 2],
    [7, 3],
    [6.5, 2.5],
    [8, 3.5],
    [7.5, 2.8]
])

# Dogs: short whiskers, very flappy ears
dogs = np.array([
    [2, 8],
    [3, 7],
    [2.5, 8.5],
    [3.5, 7.2],
    [2.8, 8]
])

# Combine data
X = np.vstack((cats, dogs))
y = np.array([0]*len(cats) + [1]*len(dogs))  # 0 = cat, 1 = dog

# Train linear regression
model = LinearRegression()
model.fit(X, y)

# Get line parameters
m = -model.coef_[0]/model.coef_[1]  # slope of separating line
c = (0.5 - model.intercept_)/model.coef_[1]  # intercept for y=0.5 boundary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset: [whisker_length, ear_flappiness]
# Cats: long whiskers, less flappy ears
cats = np.array([
    [6, 2],
    [7, 3],
    [6.5, 2.5],
    [8, 3.5],
    [7.5, 2.8]
])

# Dogs: short whiskers, very flappy ears
dogs = np.array([
    [2, 8],
    [3, 7],
    [2.5, 8.5],
    [3.5, 7.2],
    [2.8, 8]
])

# Combine data
X = np.vstack((cats, dogs))
y = np.array([0]*len(cats) + [1]*len(dogs))  # 0 = cat, 1 = dog

# Train linear regression
model = LinearRegression()
model.fit(X, y)

# Get line parameters
m = -model.coef_[0]/model.coef_[1]  # slope of separating line
c = (0.5 - model.intercept_)/model.coef_[1]  # intercept for y=0.5 boundary

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Plotting
plt.scatter(cats[:,0], cats[:,1], color='blue', label='Cats')
plt.scatter(dogs[:,0], dogs[:,1], color='red', label='Dogs')

# Plot decision boundary line
x_vals = np.linspace(0, 10, 100)
y_vals = m * x_vals + c
plt.plot(x_vals, y_vals, '--', color='green', label='Decision Boundary')

plt.xlabel("Whisker Length")
plt.ylabel("Ear Flappiness")
plt.legend()
plt.grid(True)
plt.show()

# Test a new point
test_point = np.array([[5, 5]])  # whisker length=5, ear flappiness=5
pred = model.predict(test_point)[0]
if pred >= 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Plotting
plt.scatter(cats[:,0], cats[:,1], color='blue', label='Cats')
plt.scatter(dogs[:,0], dogs[:,1], color='red', label='Dogs')

# Plot decision boundary line
x_vals = np.linspace(0, 10, 100)
y_vals = m * x_vals + c
plt.plot(x_vals, y_vals, '--', color='green', label='Decision Boundary')

plt.xlabel("Whisker Length")
plt.ylabel("Ear Flappiness")
plt.legend()
plt.grid(True)
plt.show()

# Test a new point
test_point = np.array([[5, 5]])  # whisker length=5, ear flappiness=5
pred = model.predict(test_point)[0]
if pred >= 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")
