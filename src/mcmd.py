import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MCMD():

    def __init__(self) -> None:
        """
        <MCMD fullform>\n
        Use run() method to run the algorithm
        """
        pass

    def run(self, datasets: list[pd.DataFrame], rank: int, classification_object: str = 'date', custom_beta: list[float] | None = None, e: float = 1e-10, n_iter: int = 4000, eps: int = 2000, min_pts: int = 4, delta: list[float] = [0.2, 0.01], sparsity: float = 0.00001, alpha: float | None = None, n_views_outlier: int = 1, y_lim: tuple = (0,1)) -> dict:
        """
        Run the algorithm with given paramteres.

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

        """
        self.y_lim = y_lim
        self.classification_object = classification_object
        return self._mcmd(datasets, rank, custom_beta, e, n_iter=n_iter, eps=eps, MinPts=min_pts, delta=delta, Sparsity=sparsity, alpha=alpha, n_views_outlier=n_views_outlier)

    def _mcmd(self, Datasets, rank, custom_beta, e, n_iter, eps = 1000 , MinPts = 4, delta = [0.2, 0.01], Sparsity = 0.00001, alpha = None,  n_views_outlier = 1):
        if alpha == None:
            alpha = len(Datasets)/np.sqrt(rank)
        print('Stage 1 - Data Structuring (Started) .....')
        dict1 = self._data_structuring_Multi_view_datasets(Datasets, custom_beta)
        print('Stage 1 - (Completed) ')
        print('Stage 2 - Muti-view Feature Engineering (MUNMF) (Started) .....')
        _, A, B, Results_convergance =  self._MUNMF(Datasets, dict1, rank, e, n_iter, alpha, Sparsity)
        print('Stage 2 - (Completed) ')
        print(' ')
        print('Stage 3 - Shape based outlier detection (Started) .....')
        outliers_shape_df = self._get_shape_outliers(A, B, dict1)
        outliers_shape_df1 = outliers_shape_df.groupby('date')['scale_outlier_data'].count().reset_index()
        SI_outlier_objects = outliers_shape_df1[outliers_shape_df1['scale_outlier_data'] >= n_views_outlier]['date'].unique()
        print('               Following ' + str(len (SI_outlier_objects)) + '/' + str(A.shape[0]) + f" {self.classification_object}s were classified as scale based outliers.....") 
        for outlinks in SI_outlier_objects:
            print('                     ' + outlinks) 
        A_df = pd.DataFrame((A).round(10), index = dict1[0][5] )
        scale_invariant_cluster = A_df.idxmax(axis = 1).reset_index()
        scale_invariant_cluster.loc[scale_invariant_cluster['date'].isin(SI_outlier_objects), 0] = -2
        print('Stage 3 - (Completed) ')
        print(' ')
        print('Stage 4 - Defining concensus Scaled Basis Matrix (Started) .....')
        Scale = {}
        for i in B:
            Scale[i] = np.diag(np.sqrt(np.power(pd.DataFrame(B[i]),  2).sum()))
        scaled_A = {}
        for i in B:
            scaled_A[i] = (A.dot(Scale[i])).T
            print('               Defining scale matrices (Completed)') 
        for indexo,i in enumerate(scaled_A):
            if indexo ==0:
                Consenus_scaled_A = pd.DataFrame(scaled_A[i])
            if indexo > 0:
                temp_csa = pd.concat([Consenus_scaled_A, pd.DataFrame(scaled_A[i])])
                Consenus_scaled_A = temp_csa.reset_index(drop=True)
        Consenus_scaled_A.columns = dict1[0][5]
        Consenus_scaled_A = Consenus_scaled_A.T
        Consenus_scaled_A = Consenus_scaled_A[~Consenus_scaled_A.index.isin(SI_outlier_objects)]
        print('Stage 4 - (Completed) .....')
        print('  ')
        print('Stage 5 - Scale-variant Classification: DSSIM-OPTICS (Started) .....')
        Results = self._Optics_final(Consenus_scaled_A, eps = eps , MinPts = MinPts).reset_index()
        print('Output chart stored in chart.png')
        print('Stage 5 - (Completed) .....')
        font = {'family' : 'Times New Roman',
                'weight' : 'normal',
                'size'   : 12}
        plt.rc('font', **font)
        lables_scalevariant = self._get_clust(Results, delta )
        scale_invariant_cluster.columns = ['date', 'cluster_id_SI']
        Overall_clustering = scale_invariant_cluster.merge(lables_scalevariant, on = 'date', how = 'left')
        Overall_clustering['cluster_id_SV'] = Overall_clustering['cluster_id_SV'].replace(np.nan, -2)
        Overall_clustering['cluster_id_SV']  = Overall_clustering['cluster_id_SV'].replace('outlier', -1) 
        labels_SI_ = np.array(Overall_clustering['cluster_id_SI'].astype('int64').tolist())
        labels_SV_ = np.array(Overall_clustering['cluster_id_SV'].astype('int64').tolist())
        
        # return (A, B, Consenus_scaled_A, Overall_clustering, labels_SI_, labels_SV_, outliers_shape_df1, Results_convergance)

        res_dict = {}
        res_dict['A'] = A
        res_dict['B'] = B
        res_dict['consensus_scaled_A'] = Consenus_scaled_A
        res_dict['overall_clustering'] = Overall_clustering
        res_dict['labels_SI'] = labels_SI_
        res_dict['labels_SV'] = labels_SV_
        res_dict['outliers_shape'] = outliers_shape_df1
        res_dict['results_convergance'] = Results_convergance

        return res_dict
    
    def _data_structuring_Multi_view_datasets(self, Datasets, custom_beta = None):
        dict1 = {}
        common_indices = Datasets[0].index
        for dataset2 in Datasets:
            common_indices = common_indices.intersection(dataset2.index)
        for dataset_number in range(0, len(Datasets)):
            Datasets[dataset_number] = Datasets[dataset_number].loc[common_indices]
        if custom_beta == None:
            for index, dataset in enumerate (Datasets):
                dict1[index] = self._Data_Structuring(dataset)
        if custom_beta != None:
            for index, dataset in enumerate (Datasets):
                dict1[index] = self._Data_Structuring(dataset, custom_beta[index])
        return(dict1)
    
    def _get_shape_outliers(self, A, B, dict1):
        A_df = pd.DataFrame((A).round(10), index = dict1[0][5] )
        scale_invariant_cluster = A_df.idxmax(axis = 1).reset_index()
        flago = 0
        for i in dict1.keys():
            reconstructed = pd.DataFrame(dict1[i][3] * A.dot(B[i].T), index = dict1[0][5])
            Original_df = pd.DataFrame(dict1[i][3] * dict1[i][2], index = dict1[0][5])
            mean_error_df = abs(Original_df - reconstructed).mean(axis = 1).reset_index()#abs(Volume - reconstructed).mean(axis = 1).reset_index()
            mean_error_df.columns= ['date', 'mean_error']
            cluster_members = scale_invariant_cluster.copy(deep = True).merge(mean_error_df, on = 'date', how = 'left')
            quantile25 = cluster_members.groupby(0)['mean_error'].quantile(.25).reset_index()
            quantile25.columns = [0, '25']
            quantile75= cluster_members.groupby(0)['mean_error'].quantile(.75).reset_index()
            quantile75.columns = [0, '75']
            cluster_members = cluster_members.merge(quantile25, on = 0, how = 'left').merge(quantile75, on = 0, how = 'left')
            cluster_members['IQR'] = cluster_members['75'] - cluster_members['25']
            cluster_members['outlier_score'] = cluster_members['75'] + 1.5 * cluster_members['IQR']
            cluster_members['category'] = np.where(cluster_members['mean_error'] < cluster_members['outlier_score'], 'cluster_member', 'Scale invariant Outlier')
            cluster_members[0] = cluster_members[0]+1
            cluster_members['scale_outlier_data'] = i
            cluster_members = cluster_members[cluster_members['category'] == 'Scale invariant Outlier']
            if flago == 1:
                cluster_members_concat = pd.concat([cluster_members_concat, cluster_members], ignore_index = True)
            if flago ==0:
                cluster_members_concat = cluster_members.copy()
                flago = 1
        return(cluster_members_concat)
    
    def _MUNMF(self, Datasets, dict1, rank, e, n_iter, alpha = 0.01, Sparsity = 0.00001):    
        initial = time.time()
        np.random.seed(42)
        # Generate A using the random seed
        A = np.random.rand(dict1[0][0].shape[0], rank)/100
        # Generate B using the random seed
        B = {}
        for index, i in enumerate(Datasets):
            B_temp = np.random.rand(dict1[index][0].shape[1],rank)/100
            B[index] = B_temp
        Results_convergance =[]
        error = 0
        Identity = np.identity(rank)
        print('               Initial reconstruction errors')
        for i in B:
            RMSE= self._rmse(dict1[i][4],dict1[i][3]* A.dot(B[i].T),dict1[i][0])
            error =error+ RMSE * RMSE
            print('                   rmse Dataset iteration 0', " :", round( RMSE, 2), ((Identity - A.T.dot(A)) * (Identity - A.T.dot(A))).sum())
        
        for iteration in tqdm(range(n_iter), desc="Processing"):
            if iteration%100 == 0:
                #print(error, alpha* A.sum().sum())
                error = 0
                for i in B:
                    RMSE= self._rmse(dict1[i][4],dict1[i][3]* A.dot(B[i].T),dict1[i][0])
                    error =error+ RMSE * RMSE
            error = error + alpha* A.sum().sum() * A.sum().sum() + alpha * ((Identity - A.T.dot(A)) * (Identity - A.T.dot(A))).sum()
            tempro = [iteration, error, RMSE ]
            Results_convergance.append(tempro)
            neum = np.array(pd.DataFrame((A+e)/(A+e)).replace(1.0,0))
            denom = np.array(pd.DataFrame((A+e)/(A+e)).replace(1.0,0))
            
            for i in B:
                neum = neum + ((dict1[i][0] * dict1[i][2]).dot(B[i])) 
                denom = denom+ (dict1[i][0] * (A.dot(B[i].T))).dot(B[i])
            A = A * (neum + e + (alpha * ((Identity).dot(A.T))).T )/(denom+e + Sparsity * A + (alpha * ((A.T.dot(A))).dot(A.T)).T)
            for i in B:
                B[i] = B[i] * ((dict1[i][0].T * dict1[i][2].T).dot(A) + e )/((dict1[i][0].T * B[i].dot(A.T)).dot(A) + e + Sparsity * B[i])    
            time.sleep(0.0001)
        
        temp = []
        print('               final reconstruction errors')
        for i in B:
            RMSE= self._rmse(dict1[i][4],dict1[i][3]* A.dot(B[i].T),dict1[i][0])
            error =error+ RMSE * RMSE
            print('                   rmse Dataset', iteration, " :", round( RMSE, 2), ((Identity - A.T.dot(A)) * (Identity - A.T.dot(A))).sum())
        for i in B:
            temp.append(self._rmse(dict1[i][4],dict1[i][3]* A.dot(B[i].T),dict1[i][0]))
        temp.append(((Identity - A.T.dot(A)) * (Identity - A.T.dot(A))).sum())
        return(temp, A, B, Results_convergance)
    
    def _Optics_final(self, B_hat0, eps = .1 , MinPts = 5):
        time1 = time.time()
        SSIM_Mat = []
        for ind in B_hat0.index:
            temp_ssim1 = []
            vecA= (B_hat0[B_hat0.index == ind]).iloc[0]
            for ind2 in B_hat0.index:
                vecB = (B_hat0[B_hat0.index == ind2]).iloc[0]
                temp_ssim1.append(self._get_ssim(vecA, vecB))
            SSIM_Mat.append(temp_ssim1)
        SSIM_Mat_df = pd.DataFrame(SSIM_Mat, index = B_hat0.index, columns = B_hat0.index)
        SSIM_dist = 1-(1+pd.DataFrame(SSIM_Mat_df))/2
        Visited_nodes = []
        random_initial = SSIM_dist.index[0]
        seed_list = SSIM_dist[SSIM_dist.index == random_initial].T.sort_values(by = random_initial)
        seed_list.index.name = None
        seed_list.columns= ['val']
        seed_list = seed_list[seed_list['val'] <= eps ]
        current_obj = random_initial
        item = [current_obj, 1.0]
        Visited_nodes.append(item)
        Visited_nodes_df = pd.DataFrame(Visited_nodes)
        seed_list = seed_list[seed_list.index!= current_obj]
        core_radius = seed_list.iloc[0: MinPts-1]['val'].max()
        seed_list['core'] = core_radius
        seed_list.loc[seed_list['val']<= core_radius, 'val'] = core_radius
        del seed_list['core']
        seed_list[seed_list['val'] == core_radius].index
        # print('seed_list')
        new_point = seed_list.index[0]
        while True:
            try:
                if Visited_nodes_df.shape[0] == B_hat0.shape[0]:
                    break
                # print('yo', Visited_nodes_df.shape[0])
                time2 = time.time()
                if time2 - time1 >= 600:
                    break
                new_point = seed_list.index[0]
                item = [new_point, (seed_list[seed_list.index == new_point]).T[new_point].iloc[0]]
                Visited_nodes.append(item)
                Visited_nodes_df = pd.DataFrame(Visited_nodes)
                #print(new_point)
                new_list = SSIM_dist[SSIM_dist.index == new_point].T.sort_values(by = new_point)
                new_list = new_list[~new_list.index.isin(Visited_nodes_df[0])]
                new_list = new_list[new_list.index != new_point]
                new_list.columns= ['val']
                new_list = new_list[new_list['val'] <=eps]
                core_radius_new = new_list.iloc[0: MinPts-1]['val'].max()
                new_list['core'] = core_radius_new
                new_list.loc[new_list['val']<= core_radius_new, 'val'] = core_radius_new
                del new_list['core']
                seed_list = seed_list[seed_list.index!=new_point]
                #print('seed_list', seed_list.shape, print(seed_list))
                seed_list = pd.concat([seed_list, new_list], axis=0)
                seed_list = seed_list.sort_values(by = 'val')
                seed_list = seed_list[~seed_list.index.duplicated(keep='first')]
                #print('seed_list')
            except:
                if Visited_nodes_df.shape[0] == SSIM_dist.shape[0]:
                    break
                if Visited_nodes_df.shape[0] != SSIM_dist.shape[0]:
                    while True:
                        #print('yeah')
                        SSIM_dist = SSIM_dist[~SSIM_dist.index.isin(list(Visited_nodes_df[0]))]
                        SSIM_dist = SSIM_dist.T[~SSIM_dist.T.index.isin(list(Visited_nodes_df[0]))]
                        if SSIM_dist.shape[0] ==0:
                            break
                        random_initial = SSIM_dist.index[0]
                        #print('exception', random_initial)
                        #print(random_initial)
                        seed_list = SSIM_dist[SSIM_dist.index == random_initial].T.sort_values(by = random_initial)
                        seed_list.index.name = None
                        #print(seed_list)
                        seed_list.columns= ['val']
                        seed_list = seed_list[seed_list['val'] <= eps ]
                        current_obj = random_initial
                        item = [current_obj, 1.0]
                        #print('No')
                        Visited_nodes.append(item)
                        Visited_nodes_df = pd.DataFrame(Visited_nodes)
                        #print(print(seed_list))
                        if seed_list.shape[0] == 1:
                            #print('bad case', seed_list)
                            flago= 1
                        if seed_list.shape[0] > 1:  
                            seed_list = seed_list[seed_list.index!= current_obj]
                            #print(seed_list)
                            core_radius = seed_list.iloc[0: MinPts-1]['val'].max()
                            #print(core_radius)
                            seed_list['core'] = core_radius
                            seed_list.loc[seed_list['val']<= core_radius, 'val'] = core_radius
                            del seed_list['core']
                            seed_list[seed_list['val'] == core_radius].index
                            new_point = seed_list.index[0]
                            #print (seed_list.shape[0])
                            if seed_list.shape[0] != 0:
                                break
                    pass
        Visited_nodes_df = Visited_nodes_df.set_index(0)
        # Visited_nodes_df.plot.bar(figsize = (20, 5), color = 'r')
        return(Visited_nodes_df)
    
    def _Data_Structuring(self, X, beta = None):
        W = np.array((X - X + 1).replace(np.nan, 0))
        RW = np.array((X - X).replace(np.nan, 1))
        Y = np.array(X.replace(np.nan, 0))
        if beta == None:
            beta = 1/np.sqrt((Y* Y).sum().sum())
        dict_temp = {}
        parms = [W, RW, Y * beta, 1/beta, Y, X.index]
        for i in [0,1,2,3, 4, 5]:
            dict_temp[i] = parms[i]
        return (dict_temp)
    
    def _get_clust(self, Visited_nodes_df, delta, plot = True):
        cluster_list = []
        out = Visited_nodes_df.copy(deep = True)
        out[2]= out[1]
        out.loc[out[1] < delta[0], 2  ] = np.nan
        out.loc[out[1] >= delta[0], 2  ] = 1.0
        out = out.replace(np.nan, 0)
        out[3] = (out[2]*2).cumsum()
        out['val'] = 1
        clusters = out.merge(out.groupby(3)['val'].sum().reset_index(), on = 3, how = 'left')
        clusters = clusters[[0,1,2,3,'val_y']]
        clusters.columns = ['date', 'ssim', 'modified', 'cluster_number', 'n_members' ]
        clusters['cluster_id_SV'] = clusters['cluster_number']
        clusters.loc[clusters['n_members'] == 1, 'cluster_id_SV'] = -1
        clusters['date'] =  clusters['date'].astype('str')
        list1 = clusters['cluster_id_SV'].sort_values().unique()
        list2 = []
        xx = 'outlier'
        xxx = 1
        for i in list1:
            if i == -1:
                list2.append(xx)
            if i>=0:
                list2.append(xxx)
                xxx= xxx+1
        for indexo, i in enumerate(list1):
            #print(i, list2[indexo])
            clusters['cluster_id_SV'] = clusters['cluster_id_SV'].replace(i, list2[indexo])
        if plot == True:
            fig, ax = plt.subplots(figsize=(20,8), dpi = 600)
            for jkl in delta:
                plt.axhline(y=jkl, color='black', linestyle='--')
            sns.barplot(data =clusters,x=pd.to_datetime(clusters['date'],dayfirst=True, format='mixed').dt.strftime('%d-%m'),
                        y='ssim',hue='cluster_id_SV',dodge=False, ax = ax)
            xtiks = plt.xticks(rotation=90)

            plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1, title = 'Cluster Id (SV)', fontsize =14)
            plt.rcParams['legend.title_fontsize'] = 20
            ax.set_ylim(self.y_lim)
            plt.xlabel(f"{self.classification_object.capitalize()}s", fontsize = 20)
            plt.ylabel('Reachability score', fontsize = 20)
            plt.savefig("chart.png")
        return (clusters)
    
    def _rmse(self, a,b, w):
        return np.sqrt(np.power(w*(a-b), 2).sum().sum()/(w.sum()))  
    
    def _get_ssim(self, vec_1, vec_2, **kwargs):
        if type(vec_1) != type(pd.Series(dtype = int)) or type(vec_2) != type(pd.Series(dtype = int)):
            raise TypeError("Input vectors should be pandas series dataypes")
        try:
            alpha = kwargs['alpha']
        except:
            alpha = 1
            #print("Alpha value not provided, using default value (1)")
        try:
            beta = kwargs['beta']
        except:
            beta = 1
            #print("Beta value not provided, using default value (1)")
        try:
            gamma = kwargs['gamma']
        except:
            gamma = 1
            #print("Gamma value not provided, using default value (1)")
        c_1 = 0.0
        c_2 = 0.0
        c_3 = 0.0
        l = (2 * (vec_1.mean() * vec_2.mean()) + c_1) / (vec_1.mean() ** 2 + vec_2.mean()**2 + c_1)
        c = (2 * np.sqrt(vec_1.var()) * np.sqrt(vec_2.var()) + c_2) / (vec_1.var() + vec_2.var() + c_2)
        s = (((vec_1 - vec_1.mean()) * (vec_2 - vec_2.mean())).sum()/(vec_1.shape[0]-1) + c_3) / (np.sqrt(vec_1.var()) * np.sqrt(vec_2.var()) + c_3 )
        #print(l, c, s)
        #print(((2 * vec_1.mean() * vec_2.mean())*(2 *  vec_1.cov(vec_2))) / ((vec_1.mean() **2 + vec_2.mean() ** 2) * (vec_1.var() ** 2 + vec_2.var()) ))
        return (l ** alpha) * (c ** beta) * (s ** gamma)