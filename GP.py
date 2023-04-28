import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

"""Generate Data"""
def generate_time_series_data():
    t = np.linspace(0, 10, 100)
    y = np.sin(t) + np.random.normal(0, 0.1, 100)
    return torch.tensor(t).float().view(-1, 1), torch.tensor(y).float().view(-1, 1)

"""Define Model"""
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='rbf'):
        super(GPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel == 'matern':
            base_kernel = gpytorch.kernels.MaternKernel()
        elif kernel == 'linear':
            base_kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError('Invalid kernel specified.')

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def main(kernel='rbf'):
    train_x, train_y = generate_time_series_data()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x.squeeze(), train_y.squeeze(), likelihood, kernel)

    """Train"""
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iterations = 50
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y).sum()
        loss.backward()
        optimizer.step()

    """Predict"""
    model.eval()
    likelihood.eval()

    test_x = torch.linspace(0, 12, 120).view(-1, 1)
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))

    """Plot"""
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy().squeeze(),
                        observed_pred.confidence_region()[0].numpy(),
                        observed_pred.confidence_region()[1].numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Predicted Mean', 'Confidence Interval'])
        ax.set_title(f'Gaussian Process Regression with {kernel.upper()} Kernel')
        plt.show()

if __name__ == '__main__':
    # Choose different kernel functions: 'rbf', 'matern', 'linear' ...
    main(kernel='rbf')
