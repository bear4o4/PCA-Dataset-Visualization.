# PCA-Dataset-Visualization.

This repository contains Python classes for handling 2D and 3D datasets using NumPy arrays. It includes methods for loading data, displaying data shapes, plotting data, and applying Principal Component Analysis (PCA) to reduce the dimensionality of the 3D dataset.

## Usage

## Dataset2D Class
The Dataset2D class handles 2D datasets.  
Methods
load_data(x_file, y_file): Loads x and y data from .npy files.
display_shape(): Prints the shapes of the x and y arrays.
plot_data(): Creates a 2D scatter plot of x vs y using Matplotlib.

## Dataset3D Class
The Dataset3D class inherits from Dataset2D and handles 3D datasets.
Additional Methods
load_data(x_file, y_file, z_file): Loads x, y, and z data from .npy files.
display_shape(): Prints the shapes of the x, y, and z arrays.
plot_data(): Creates a 3D scatter plot of x, y, and z using Matplotlib.
apply_pca(): Applies PCA to reduce the 3D dataset to 2D, saves the reduced data, and computes the determinant and inverse of the covariance matrix.
plot_pca_data(): Creates a 2D scatter plot of the PCA-reduced data.
