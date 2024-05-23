
# MCMD ; Developed by: [Shubham Sharma](mailto:s55.sharma@hdr.qut.edu.au)

MCMD (**Multi-view Classification framework based on Consensus Matrix Decomposition**) is a multiview algorithm originally developed for clustering different days based on the similarity of their network-wide traffic states throughout the day. Since the traffic state is multifaceted and requires more than one parameter for a holistic definition, MCMD was designed with several fundamental requirements to address the limitations and challenges of real-world traffic datasets. These requirements include effectively handling the high dimensionality of various datasets, compatibility with one or more datasets, the ability to identify outliers (with context on why the day was an outlier), and handling missing information, which is a common real-world phenomenon. 

## Key Characteristics of MCMD clustering framework
1. <code style = "color : aqua"> **Single and Multi data compatibility** </code>:  The algorithm's design is compatible with both single and multiple datasets.
2. <code style = "color : aqua"> **Effectively handles High dimensional datasets** </code>:   It translates multiple datasets simultaneously into a lower-dimensional space by exploring their shared and distinct structures to address issues related to the curse of dimensionality.
3. <code style = "color : aqua"> **Obtains two types of cluster labels** </code>:  It obtains two types of clusters. The first is based on feature similarity (scale-invariant or shape similarity). The second type of class labels is based on scale-variant similarities, with cluster members holding both high shape and scale-based similarities.
4. <code style = "color : aqua"> **Can identify two types of outliers** </code>:  The algorithm can identify two types of outliers. The first are those with highly dissimilar shapes compared to other clusters. The second type of outliers are those which may have similar shapes to other objects but their scale is anomalous.
5. <code style = "color : aqua"> **Default hyperparameters** </code>: The only necessary hyperparameter to tune is the "rank of factorization". Other hyperparameters are assigned default values using the Normalized Weighted Sum (NWS) Approach. However, the algorithm's design allows flexibility for the implementation and tuning of other hyperparameters as well.
6. <code style = "color : aqua"> **Highly Robust to missing information** </code>:  The algorithm does not require any missing data imputation. It has inherent missing data imputation properties derived from matrix factorization principles and enhanced by the capability to share information across datasets (or views).

## Working Principle

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0968090X24001281-gr2_lrg.jpg" alt="MCMD Framework: https://ars.els-cdn.com/content/image/1-s2.0-S0968090X24001281-gr2_lrg.jpg " width="600"> </img> 

**Figure 1**: ***MCMD Framework (Source: https://doi.org/10.1016/j.trc.2024.104607 (Fig. 2))*** 

Five key sequential modules in MCMD Framework are:
1. <code style = "color : aqua"> **Data Structuring** </code>:Structure all datasets as different matrices. Each matrix should have the objects to be clustered along rows, i.e., each row represent a unique object to be clustered (days in the original study). The number of objects andir sequencing along rows in all the matrices should be identical. 
2. <code style = "color : aqua"> **MUNMF Algorithm** </code>: This algorithm is based on a coupled non-negative matrix decomposition principle, with orthogonality constraints on the Basis matrix (matrix defined for rows). 
3. <code style = "color : aqua"> **Outlier Removal Module** </code>: Identifies the shape-dissimilarity based outliers.
4. <code style = "color : aqua"> **Define Concatenated Scaled Basis Matrix** </code>: Joins different datasets (views) in their lower-dimensional representation.
5. <code style = "color : aqua"> **DSSIM-OPTICS module** </code>: A density-based clustering algorithm, i.e., OPTICS to mine clusters with members exhibiting both shape and scale-similarities. DSSIM is a structural dissimilarity index to quantify similarity between two objects.

# Reference

The original article is published in **Transportation Research Part-C**. To refer the article, use: 

***Sharma, Shubham, Richi Nayak, and Ashish Bhaskar. "Multi-view feature engineering for day-to-day joint clustering of multiple traffic datasets." Transportation Research Part C: Emerging Technologies 162 (2024): 104607. https://doi.org/10.1016/j.trc.2024.104607*** 


Parameters
-----------
* <code style = "color : aqua"> **datasets** </code>:  A list of one or more datasets to be jointly clustered, inputted in pandas data frame format with common indices.
* <code style = "color : aqua"> **rank** </code>: Rank of factorization, i.e., the number of Scale-Invariant clusters to be mined in the multi-view datasets.
* <code style = "color : aqua"> **classification_object** </code>: An index common in all datasets on the basis of which classification is to be done. For example: date
* <code style = "color : aqua"> **custom_beta** </code>: A list of weightages assigned to each dataset within "Datasets" in the feature engineering stage. The length of "Datasets" and custom_beta should be the same.
* <code style = "color : aqua"> **e** </code>: A small number, default value is 1e-07.
* <code style = "color : aqua"> **n_iter** </code>: Number of iterations in the feature engineering process (MUNMF).
* <code style = "color : aqua"> **eps** </code>: Epsilon parameter of OPTICS, default value is set at 2000.
* <code style = "color : aqua"> **min_pts** </code>: min_pts parameter of OPTICS, i.e., the minimum number of points for a group of points to be considered a cluster.
* <code style = "color : aqua"> **delta** </code>: delta is a list of reachability thresholds (DSSIM metric that ranges from 0 to 1) for mining variable density clusters in the data.
* <code style = "color : aqua"> **sparsity** </code>: sparsity is the weightage of the L2 norm for different factor matrices in MUNMF.
* <code style = "color : aqua"> **alpha** </code>: alpha is the weight parameter for orthogonality constraint in MUNMF. Default value is set at "number of views (or dataframes in Datasets) / rank.
* <code style = "color : aqua"> **n_views_outlier** </code>: Minimum number of views in which an object should be classified as a "shape-based outlier" to be considered as an overall "shape-based" outlier in multi-view classification
* <code style = "color : aqua"> **y_lim** </code>: range of y-axis in chart. Default is (0,1), i.e. 0 to 1

Outputs (dictionary)
--------
* <code style = "color : aqua"> **A** </code>: basis matrix
* <code style = "color : aqua"> **B** </code>:  A list of coefficient matrices. The number of coefficient matrices in B is equal to the number of dataframes in Datasets. Each B represents the latent representation of each view in the transformed space.
* <code style = "color : aqua"> **consensus_scaled_A** </code>: Consensus scaled basis matrix.
* <code style = "color : aqua"> **overall_clustering** </code>:  A data frame that contains various calculations performed in the clustering process.
* <code style = "color : aqua"> **labels_SI** </code>: Scale Invariant (SI) classification labels. -2 indicates a shape-based outlier.
* <code style = "color : aqua"> **labels_SV** </code>: Scale-Variant (SV) Classification labels. -1 indicates a scale-based outlier.
* <code style = "color : aqua"> **outliers_shape** </code>: Dataframe comprising information about which object was classified as an outlier in which view.
* <code style = "color : aqua"> **results_convergence** </code>: Reconstruction errors for individual datasets and orthogonality condition.



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

## Are you facing any issues with implementation of MCMD? Did you spot any bug in the code?
Feel free to send an email at s50.sharma@qut.edu.au with subject : "**MCMD implementation issue**". We'll be more than happy to help you out.

