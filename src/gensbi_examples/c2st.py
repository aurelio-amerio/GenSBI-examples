from typing import Optional

import jax
from jax import numpy as jnp
from jax import Array
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X: Array,
    Y: Array,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> Array:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = jnp.mean(X, axis=0)
        X_std = jnp.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std


    if noise_scale is not None:
        key = jax.random.PRNGKey(seed)
        X += noise_scale * jax.random.normal(key, X.shape) * noise_scale
        Y += noise_scale * jax.random.normal(key, Y.shape) * noise_scale


    # Convert to numpy if not already

    X = np.asarray(X)
    Y = np.asarray(Y)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return scores


def c2st_auc(
    X: Array,
    Y: Array,
    seed: int = 1,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    ) -> Array:
    """Classifier-based 2-sample test returning AUC (area under curve)

    Same as c2st, except that it returns ROC AUC rather than accuracy

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    Returns:
        Metric
    """
    return c2st(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        scoring="roc_auc",
        z_score=z_score,
        noise_scale=noise_scale,
    )