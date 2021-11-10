import numpy as np
import datetime
import umap
import matplotlib.pyplot as plt
import os
import pickle
import time
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from matplotlib.ticker import NullFormatter

print('setting up logging...')
dt = datetime.datetime.fromtimestamp(time.time())
logdir = os.path.join('./outputs/' ,dt.strftime('%Y-%m-%d_%H:%M:%S'))


print(f'Logging to {logdir}')
if not os.path.exists(logdir):
    os.makedirs(logdir)

    
def construct_datasets(n_samples):
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    return [
        make_moons(n_samples=n_samples, noise=0.3, random_state=0),
        make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]

datasets = construct_datasets(100)
names = ['moons', 'circles', 'linearly seperable']


def plot_umap_hyperparam_sweep(ds, name, min_dists, n_neighbors, umap_seed, verbose=False):
    fig = plt.figure(figsize=(27, 9))
    k = 1
    X, y = ds
    ax = plt.subplot(len(n_neighbors), len(min_dists) + 1,  k)
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax.set_title("input data")
    ax.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=cm_bright, edgecolors="k")
    k += 1
    for j, n in enumerate(n_neighbors):
        for i, min_dist in enumerate(min_dists):
            if k % (len(min_dists) + 1) == 1: # don't plot below input data subplot
                k += 1
            ax = plt.subplot(len(n_neighbors), len(min_dists) + 1, k)

            umapper = umap.UMAP(random_state=umap_seed, min_dist=min_dist, n_neighbors=n, verbose=verbose, n_epochs=10000, log_losses="after",)
            umap_proj = umapper.fit_transform(X)

            ax.scatter(umap_proj[:, 0], umap_proj[:, 1], c=y, cmap=cm_bright, edgecolors="k")
            ax.set_title(f' n_neighbors={n}, min_dist={min_dist}' )
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            k+=1
        
    fig.suptitle(f'rand_seed={umap_seed}')
    fig.savefig(join(logdir, f'umap_hyperparams_{name}_{len(n_neighbors)}n_{len(min_dists)}_md'))
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.show()
     

for name, ds in zip(names, datasets):
#     plot_umap_hyperparam_sweep(ds, name, np.arange(0.01, 0.5, 0.01), np.arange(5, 30, 5), 42)
    plot_umap_hyperparam_sweep(ds, name, [0.1], [2], 42, True)

    break



