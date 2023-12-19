
# MCMD

MCMD (Multi-view Classification framework based on Consensus Matrix Decomposition) is a Multiview algorithm for a joint Scale-Invariant (SI) and Scale-Variant (SV) classification of multiple related datasets. The algorithm is computationally robust to missing information (NaN) in the input datasets, can cluster both single and multiple datasets together, finds clusters based on how similar the shapes of classification objects are (SI classification), identifies the objects which have anomalous shapes and classifies them as "shape-based outliers," mines clusters having both shape and scale similarity, and classifies the scale-based outliers in the dataset. The algorithm is based on a Multi-view Uni-orthogonal Non-negative Matrix Factorization (MUNMF) algorithm combined with an OPTICS-based algorithm.

Developed by: [Shubham Sharma](mailto:s55.sharma@hdr.qut.edu.au)


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
    rank = 8
    custom_beta = [1,2]
    delta = [0.1, 0.02]
    n_iter = 4000
    n_views_outlier = 2
)
```

## Dependencies
* numpy
* pandas
* matplotlib
* seaborn
* tqdm