# Let's import all the necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from gurobipy import GRB
import gurobipy as gp
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from itertools import product
import matplotlib.pyplot as plt
import numpy as np


# Create and deploy the optimization model

# NOTE: This function assumes the design matrix features does not contain
# a column for the intercept

def miqp(features, response, non_zero, warm_up=None, verbose=False):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    regressor = gp.Model()
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    # Append a column of ones to the feature matrix to account for the y-intercept
    X = np.concatenate([features, np.ones((samples, 1))], axis=1)

    # Decision variables
    beta = regressor.addVars(dim + 1, lb=-GRB.INFINITY, name="beta")  # Weights
    intercept = beta[dim]  # Last decision variable captures the y-intercept
    intercept.varname = 'intercept'
    # iszero[i] = 1 if beta[i] = 0
    iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

    # Objective Function (OF): minimize 1/2 * RSS using the fact that
    # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
    Quad = np.dot(X.T, X)
    lin = np.dot(response.T, X)
    obj = sum(0.5 * Quad[i, j] * beta[i] * beta[j]
              for i, j in product(range(dim + 1), repeat=2))
    obj -= sum(lin[i] * beta[i] for i in range(dim + 1))
    obj += 0.5 * np.dot(response, response)
    regressor.setObjective(obj, GRB.MINIMIZE)

    # Constraint sets
    for i in range(dim):
        # If iszero[i]=1, then beta[i] = 0
        regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
    regressor.addConstr(iszero.sum() == dim - non_zero)  # Budget constraint

    # We may use the Lasso or prev solution with fewer features as warm start
    if warm_up is not None and len(warm_up) == dim:
        for i in range(dim):
            iszero[i].start = (abs(warm_up[i]) < 1e-6)

    if not verbose:
        regressor.params.OutputFlag = 0
    regressor.params.timelimit = 60
    regressor.params.mipgap = 0.001
    regressor.optimize()

    coeff = np.array([beta[i].X for i in range(dim)])
    return intercept.X, coeff

# Define functions necessary to perform hyper-parameter tuning via cross-validation


def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask, :]
    xtest = features[~train_mask, :]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest


def cross_validate(features, response, non_zero, folds, standardize, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_cv = 0
    # Exclude folds from training, one at a time,
    # to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size: min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(
            features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        intercept, beta = miqp(xtrain, ytrain, non_zero)
        ypred = np.dot(xtest, beta) + intercept
        mse_cv += mse(ytest, ypred) / folds
    # Report the average out-of-sample MSE
    return mse_cv


def L0_regression(features, response, folds=5, standardize=False, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_mse = np.inf
    best = 0
    # Grid search to find best number of features to consider
    for i in range(1, dim + 1):
        val = cross_validate(features, response, i, folds=folds,
                             standardize=standardize, seed=seed)
        if val < best_mse:
            best_mse = val
            best = i
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    intercept, beta = miqp(features, response, best)
    return intercept, beta

# Define how the bar chart should be displayed


def plot_bar_chart(performance):
    """
    Display the performance of all three models in a bar chart.
    """
    bar = plt.bar([1, 2, 3], performance, color=['r', 'g', 'y'],
                  tick_label=['OLS', 'Lasso', 'L0-Regression'])
    plt.title('Out-of-Sample MSE')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, np.floor(np.min(performance)),
              np.ceil(np.max(performance))))
    plt.show()


# Load data and split into train (80%) and test (20%)
boston = load_boston()
X = boston['data']
y = boston['target']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20,
                                                random_state=10101)

# OLS regression using all features
lr = linear_model.LinearRegression()
lr.fit(Xtrain, ytrain)
# Lasso with cross-validated penalization (lambda)
lasso = linear_model.LassoCV(cv=5)
lasso.fit(Xtrain, ytrain)
# L0-regression where the best feature subset is selected via cross-validation
intercept, beta = L0_regression(Xtrain, ytrain, seed=10101)

# Compare their performance using a bar chart
performance = []
performance.append(mse(ytest, lr.predict(Xtest)))
performance.append(mse(ytest, lasso.predict(Xtest)))
performance.append(mse(ytest, np.dot(Xtest, beta) + intercept))
plot_bar_chart(performance)

np.random.seed(10101)
num_tries = 500
best_alpha = None
best_score = -np.inf
for i in range(num_tries):
    # log-linear search for alpha in the domain [0.001, 1000]
    exponent = np.random.uniform(-3, 3)
    alpha = np.power(10, exponent)
    pipeline = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=alpha))
    scores = cross_val_score(pipeline, Xtrain, ytrain,
                             cv=5, scoring='neg_mean_squared_error')
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_alpha = alpha

# Standardize the features so they have an avg of 0 and a sample var of 1
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain_std = scaler.transform(Xtrain)
Xtest_std = scaler.transform(Xtest)

# OLS regression using all features
lr = linear_model.LinearRegression()
lr.fit(Xtrain_std, ytrain)

# Lasso with cross-validated penalization (lambda)
lasso = linear_model.Lasso(alpha=best_alpha)
lasso.fit(Xtrain_std, ytrain)
# L0-regression where the best feature subset is selected via cross-validation
intercept, beta = L0_regression(Xtrain, ytrain, standardize=True, seed=10101)

# Compare their performance using a Bar chart
performance = []
performance.append(mse(ytest, lr.predict(Xtest_std)))
performance.append(mse(ytest, lasso.predict(Xtest_std)))
performance.append(mse(ytest, np.dot(Xtest_std, beta) + intercept))
plot_bar_chart(performance)

ols_features = np.sum(np.abs(lr.coef_) >= 1e-8)
lasso_features = np.sum(np.abs(lasso.coef_) >= 1e-8)
l0_features = np.sum(np.abs(beta) >= 1e-8)
print("OLS regression kept {0} features.".format(ols_features))
print("The Lasso kept {0} features.".format(lasso_features))
print("L0-Regression kept {0} features.".format(l0_features))
