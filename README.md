# Coding tasks from "The Elements of Statistical Learning"

The repo contains a collection of coding tasks from the book "The Elements of Statistical Learning" (Trevor Hastie, Robert Tibshirani, Jerome Friedman), aimed to familiarise with some of the most common techniques used in supervised and unsupervised learning.

In this repo, the following models and methods have been applied to a variety of datasets (provided by the authors of the book).

### Supervised methods

- Linear methods for regression.
	- Ordinary least square and its regularized versions, like Ridge regression and Lasso.
	- Principal component regression.
	- Partial least square.
	- Best-subset linear regression.

- Linear methods for classification.
	- Linear Discriminant Analysis (and its Quadratic and Local version).

- Prototypes methods.
	- K-nearest-neigbours (K-NN).
	- K-NN on data with symmetries (hints in dataset, invariant metric, tangent distance).

- Trees and additive models.
	- Generalised Additive Models (GAM)
	- Classificationa and Regression Trees (CART).
	- Multivariate Adaptive Regression Splines (MARS).
	- Patient Rule Induction Methods (PRIM).

- Ensample learners.
	- AdaBoost.
	- Random Forests.

- Neural Networks.

- Projection Pursuit Regression.

### Unsupervised methods

- Clustering algorithms.
	- K-means.
	- Self-Organizing Maps (SOM).

- Principal Component Analysis (PCA).

- Non-negative Matrix Factorization.

- Association rules.

- Gaussian Graphical Models (with and without known structure).

### General techniques.

- Smoothing methods and basis expansion.
	- Splines (cubic splines, natural smoothing splines, B-splines).

- Model assessment techniques.
	- K-fold cross validation.
	- Bootstrapping.
	- Estimation of test error via AIC, BIC.

Among the packages used for completing the tasks there are [sklearn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/), [csaps](https://csaps.readthedocs.io/), and [pygam](https://pygam.readthedocs.io/).
