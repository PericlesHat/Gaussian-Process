import numpy as np
import matplotlib.pyplot as plt

# Define the kernel function
def rbf_kernel(x1, x2, length_scale=1.0, signal_variance=1.0):
    return signal_variance * np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)

# Compute the kernel matrix
def compute_kernel_matrix(X, kernel_func):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j])
    return K

# GP model training
def train_gp(X_train, y_train, kernel_func):
    K = compute_kernel_matrix(X_train, kernel_func) + 1e-6 * np.eye(len(X_train))
    L = np.linalg.cholesky(K)  # Cholesky decomposition
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    return L, alpha

# GP model prediction
def predict_gp(X_test, X_train, L, alpha, kernel_func):
    K_test = np.array([kernel_func(x_test, x_train) for x_test in X_test for x_train in X_train]).reshape(len(X_test), len(X_train))
    mu = K_test.dot(alpha)
    v = np.linalg.solve(L, K_test.T)
    cov = compute_kernel_matrix(X_test, kernel_func) - v.T.dot(v)
    return mu, cov

# Example usage
X_train = np.linspace(0, 10, 100)
y_train = np.sin(X_train) + np.random.normal(0, 0.1, 100)
X_test = np.linspace(0, 12, 120)

L, alpha = train_gp(X_train, y_train, rbf_kernel)
mu, cov = predict_gp(X_test, X_train, L, alpha, rbf_kernel)

std_dev = np.sqrt(np.diag(cov))
confidence_level = 0.95
z_score = 1.96  # For a 95% confidence interval

lower_bound = mu - z_score * std_dev
upper_bound = mu + z_score * std_dev


plt.figure()
plt.plot(X_test, mu, 'k', lw=2, label='Predictive mean')
plt.fill_between(X_test[:], lower_bound, upper_bound, color='gray', alpha=0.5, label=f'{confidence_level * 100:.0f}% confidence interval')
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

