
# MCMD

MCMD (Multi-view Classification framework based on Consensus Matrix Decomposition) is a Multiview algorithm for a joint Scale-Invariant (SI) and Scale-Variant (SV) classification of multiple related datasets. The algorithm is computationally robust to missing information (NaN) in the input datasets, can cluster both single and multiple datasets together, finds clusters based on how similar the shapes of classification objects are (SI classification), identifies the objects which have anomalous shapes and classifies them as "shape-based outliers," mines clusters having both shape and scale similarity, and classifies the scale-based outliers in the dataset. The algorithm is based on a Multi-view Uni-orthogonal Non-negative Matrix Factorization (MUNMF) algorithm combined with an OPTICS-based algorithm.

Developed by: [Shubham Sharma](mailto:s55.sharma@hdr.qut.edu.au)

Parameters
-----------
* datasets :  A list of one or more datasets to be jointly clustered, inputted in pandas data frame format with common indices.
* rank: Rank of factorization, i.e., the number of Scale-Invariant clusters to be mined in the multi-view datasets.
* classification_object: An index common in all datasets on the basis of which classification is to be done. For example: date
* custom_beta: A list of weightages assigned to each dataset within "Datasets" in the feature engineering stage. The length of "Datasets" and custom_beta should be the same.
* e: A small number, default value is 1e-07.
* n_iter: Number of iterations in the feature engineering process (MUNMF).
* eps: Epsilon parameter of OPTICS, default value is set at 2000.
* min_pts: min_pts parameter of OPTICS, i.e., the minimum number of points for a group of points to be considered a cluster.
* delta: delta is a list of reachability thresholds (DSSIM metric that ranges from 0 to 1) for mining variable density clusters in the data.
* sparsity: sparsity is the weightage of the L2 norm for different factor matrices in MUNMF.
* alpha: alpha is the weight parameter for orthogonality constraint in MUNMF. Default value is set at "number of views (or dataframes in Datasets) / rank.
* n_views_outlier: Minimum number of views in which an object should be classified as a "shape-based outlier" to be considered as an overall "shape-based" outlier in multi-view classification
* y_lim: range of y-axis in chart. Default is (0,1), i.e. 0 to 1

Outputs (dictionary)
--------
* A: basis matrix
* B:  A list of coefficient matrices. The number of coefficient matrices in B is equal to the number of dataframes in Datasets. Each B represents the latent representation of each view in the transformed space.
* consensus_scaled_A: Consensus scaled basis matrix.
* overall_clustering:  A data frame that contains various calculations performed in the clustering process.
* labels_SI: Scale Invariant (SI) classification labels. -2 indicates a shape-based outlier.
* labels_SV: Scale-Variant (SV) Classification labels. -1 indicates a scale-based outlier.
* outliers_shape: Dataframe comprising information about which object was classified as an outlier in which view.
* results_convergence: Reconstruction errors for individual datasets and orthogonality condition.



## Installation

```bash
  pip install mcmd
```
    
## Usage/Examples

```py
from mcmd import MCMD

mcmd = MCMD()

res_dict = mcmd.run(
    [dataset_1, dataset_2], # datasets should have a common index, for example: dates
    rank = 8,
    custom_beta = [1,2],
    delta = [0.1, 0.02],
    n_iter = 4000,
    n_views_outlier = 2,
)
```

## Dependencies
* numpy
* pandas
* matplotlib
* seaborn
* tqdm