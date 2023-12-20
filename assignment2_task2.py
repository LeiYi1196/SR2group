import numpy as np

def initialize_parameters(data, K_components, seed):
    np.random.seed(seed)
    # Initialize random means and variances
    means = np.random.choice(data, size=K_components)
    variances = np.random.random(size=K_components)
    # Initialize weights uniformly
    weights = np.ones(K_components) / K_components
    return means, variances, weights

def expectation_maximization(data, means, variances, weights, epsilon):
    old_likelihood = 0
    convergence = False

    while not convergence:
        # Expectation Step
        responsibilities = []
        for i in range(len(data)):
            pdf_values = np.array([weights[k] * (1 / np.sqrt(2 * np.pi * variances[k])) *
                                   np.exp(-0.5 * ((data[i] - means[k]) ** 2) / variances[k]) for k in range(len(means))])
            pdf_values /= np.sum(pdf_values)
            responsibilities.append(pdf_values)

        responsibilities = np.array(responsibilities)

        # Maximization Step
        Nk = np.sum(responsibilities, axis=0)
        new_means = np.sum(data.reshape(-1, 1) * responsibilities, axis=0) / Nk
        new_variances = np.sum(responsibilities * ((data.reshape(-1, 1) - new_means) ** 2), axis=0) / Nk
        new_weights = Nk / len(data)

        # Check convergence
        likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
        if np.abs(likelihood - old_likelihood) < epsilon:
            convergence = True
        else:
            old_likelihood = likelihood

        # Update parameters
        means = new_means
        variances = new_variances
        weights = new_weights

    return means, variances, weights

def custom_GMM_uni(data, K_components, epsilon=1e-6, seed=None):
    np.random.seed(seed)
    means, variances, weights = initialize_parameters(data, K_components, seed)
    means, variances, weights = expectation_maximization(data, means, variances, weights, epsilon)

    params_dict = {
        'omega': weights,
        'mu': means,
        'Sigma': variances
    }

    return params_dict

x = np.array([4.6, 12.4, 10.2, 12.8, 12.3, 13.0, 4.3, 4.2, 9.2, 12.0,
12.6, 5.6, 4.6, 9.4, 5.6, 10.0, 10.4, 4.0, 11.4, 10.2])
parameters = custom_GMM_uni(data=x, K_components=2, epsilon=10e-6, seed=1234)
print(parameters)