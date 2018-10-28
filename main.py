import numpy as np
import pygraphviz as pgv
from scipy.stats import multivariate_normal

from src.DataLoader import DataLoader
from model.GraphLasso import GraphLasso

def adj_matrix_to_graph(matrix : np.ndarray, keys, l):
    # check whether two dimensions matches
    if matrix.shape[1] != len(keys):
        print("Error : The number of keys doesn't match with the length of a matrix")
        return -1

    graph = pgv.AGraph(directed=False)

    for i in range(len(keys)):
        graph.add_node(n=keys[i])

    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1], 1):
            if matrix[i][j] != 0:
                graph.add_edge(u=keys[i], v=keys[j])

    return graph

def Main():
    # load the data
    data = DataLoader()
    lambdas = [(i + 1) * 2.0 for i in range(32)]
    batch_size = 500
    num_batch = 2
    sigma = 1.0
    noise = 1e-3
    models = []
    keys = data.keys()

    for i in range(len(lambdas)):
        model = GraphLasso(lambdas[i],
                           batch_size=batch_size,
                           max_iter_outer=20,
                           max_iter_inner=20,
                           eps=1e-4)
        models.append(model)

    for i in range(len(models)):
        sum = np.zeros(shape=(len(keys) * len(keys),))
        list_theta = []

        for each_batch in range(num_batch):
            X_batch, y_batch = data.sample_batch(500)
            # estimate the precision matrix
            theta = models[i].estimate(X_batch)
            sum = sum + theta.flat
            list_theta.append(theta.flat)
        list_theta = np.stack(list_theta, axis=0)
        # compute the training error
        mean = sum / float(num_batch)
        """
        cov = np.cov(np.transpose(list_theta))
        cov = cov + noise * np.identity(cov.shape[0])
        dist = multivariate_normal(mean=mean, cov=cov)
        surrogate = 0.0
        for i in range(num_batch):
            log_prob = dist.logpdf(list_theta[i])
            surrogate = surrogate + log_prob
        train_error = -2 * surrogate / float(batch_size)
        df = float(len(keys) * (len(keys) - 1)) / 2.0
        AIC = train_error + 2.0 * df * sigma * sigma / float(num_batch)
        print("AIC for lambda {}: {}".format(lambdas[i], AIC))
        """

        adj_matrix = np.reshape(mean, newshape=(len(keys), len(keys)))
        # compute the adjacent matrix
        adj_matrix[np.abs(adj_matrix) < 1e-9] = 0
        adj_matrix = np.abs(np.sign(adj_matrix))
        for diag in range(adj_matrix.shape[0]):
            adj_matrix[diag, diag] = 0

        graph = adj_matrix_to_graph(adj_matrix, keys, lambdas[i])
        graph.layout(prog='circo')
        graph.draw("./graphWithLambdaValue{}.png".format(lambdas[i]))


if __name__ == "__main__":
    Main()