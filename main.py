import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Dataset2D:
    def __init__(self):
        self.x = None
        self.y = None

    def load_data(self, x_file, y_file):
        self.x = np.load(x_file)
        self.y = np.load(y_file)

    def display_shape(self):
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")

    def plot_data(self):
        plt.scatter(self.x, self.y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D Scatter Plot')
        plt.show()

class Dataset3D(Dataset2D):
    def __init__(self):
        super().__init__()
        self.z = None
        self.x_pca = None
        self.y_pca = None

    def load_data(self, x_file, y_file, z_file):
        super().load_data(x_file, y_file)
        self.z = np.load(z_file)

    def display_shape(self):
        super().display_shape()
        print(f"z shape: {self.z.shape}")

    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Scatter Plot')
        plt.show()

    def apply_pca(self):

        data = np.vstack((self.x, self.y, self.z)).T
        mean = np.mean(data, axis=0)
        centered_data = data - mean
        cov_matrix = np.cov(centered_data, rowvar=False)


        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        top_indcies=idx[:2]

        eigenvectors = eigenvectors[:, top_indcies]


        pca_data = centered_data.dot(eigenvectors[:, :2])
        self.x_pca = pca_data[:, 0]
        self.y_pca = pca_data[:, 1]


        np.save('x_pca.npy', self.x_pca)
        np.save('y_pca.npy', self.y_pca)


        det = np.linalg.det(cov_matrix)
        print(f"Determinant of covariance matrix: {det}")
        if det != 0:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            print(f"Inverse of covariance matrix:\n{inv_cov_matrix}")

    def plot_pca_data(self):
        plt.scatter(self.x_pca, self.y_pca)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA Scatter Plot')
        plt.show()



print("task1")
dataset = Dataset3D()
dataset.load_data('x_data.npy', 'y_data.npy', 'z_data.npy')


dataset.display_shape()
print("task2")
dataset_2d = Dataset2D()
dataset_2d.load_data('x_data.npy', 'y_data.npy')

dataset_3d = Dataset3D()
dataset_3d.load_data('x_data.npy', 'y_data.npy', 'z_data.npy')


dataset_2d.plot_data()
dataset_3d.plot_data()

print("task3")
dataset_3d = Dataset3D()
dataset_3d.load_data('x_data.npy', 'y_data.npy', 'z_data.npy')


dataset_3d.apply_pca()

dataset_3d.plot_pca_data()