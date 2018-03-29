from numpy import loadtxt
from numpy import sort
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import lightgbm as lgb 
import numpy as np
from boostaroota import BoostARoota
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier

var = [
    'diff_distinct_CustomerID_grad_grad_mean_non_sorted',
    'diff_distinct_CustomerID_grad_grad_std_non_sorted',
    'diff_distinct_CustomerID_kurt_non_sorted',
    'diff_distinct_CustomerID_skew_non_sorted',
    'diff_distinct_CustomerID_grad_kurt_non_sorted',
    'diff_distinct_CustomerID_grad_skew_non_sorted',
    'diff_distinct_CustomerID_grad_grad_0_non_sorted',
    'diff_distinct_CustomerID_grad_grad_10_non_sorted',
    'diff_distinct_CustomerID_grad_grad_20_non_sorted',
    'diff_distinct_CustomerID_grad_grad_30_non_sorted',
    'diff_distinct_CustomerID_grad_grad_40_non_sorted',
    'diff_distinct_CustomerID_grad_grad_50_non_sorted',
    'diff_distinct_CustomerID_grad_grad_60_non_sorted',
    'diff_distinct_CustomerID_grad_grad_70_non_sorted',
    'diff_distinct_CustomerID_grad_grad_80_non_sorted',
    'diff_distinct_CustomerID_grad_grad_90_non_sorted',
    'diff_distinct_CustomerID_grad_grad_100_non_sorted',
    'diff_distinct_CustomerID_grad_grad_kurt_non_sorted',
    'diff_distinct_CustomerID_grad_grad_skew_non_sorted',
    'diff_CustomerID_grad_grad_mean_non_sorted',
    'diff_CustomerID_grad_grad_std_non_sorted',
    'diff_CustomerID_kurt_non_sorted',
    'diff_CustomerID_skew_non_sorted',
    'diff_CustomerID_grad_kurt_non_sorted',
    'diff_CustomerID_grad_skew_non_sorted',
    'diff_CustomerID_grad_0_non_sorted',
    'diff_CustomerID_grad_10_non_sorted',
    'diff_CustomerID_grad_20_non_sorted',
    'diff_CustomerID_grad_30_non_sorted',
    'diff_CustomerID_grad_40_non_sorted',
    'diff_CustomerID_grad_50_non_sorted',
    'diff_CustomerID_grad_60_non_sorted',
    'diff_CustomerID_grad_70_non_sorted',
    'diff_CustomerID_grad_80_non_sorted',
    'diff_CustomerID_grad_90_non_sorted',
    'diff_CustomerID_grad_100_non_sorted',
    'diff_CustomerID_grad_grad_kurt_non_sorted',
    'diff_CustomerID_grad_grad_skew_non_sorted',
    
    'hour_ratio_kurt', 'hour_ratio_skew',

    'diff_distinct_CustomerID_grad_nonsorted_mean',
    'diff_distinct_CustomerID_grad_nonsorted_std',
    'diff_distinct_CustomerID_grad_nonsorted_0',
    'diff_distinct_CustomerID_grad_nonsorted_10',
    'diff_distinct_CustomerID_grad_nonsorted_20',
    'diff_distinct_CustomerID_grad_nonsorted_30',
    'diff_distinct_CustomerID_grad_nonsorted_40',
    'diff_distinct_CustomerID_grad_nonsorted_50',
    'diff_distinct_CustomerID_grad_nonsorted_60',
    'diff_distinct_CustomerID_grad_nonsorted_70',
    'diff_distinct_CustomerID_grad_nonsorted_80',
    'diff_distinct_CustomerID_grad_nonsorted_90',
    'diff_distinct_CustomerID_grad_nonsorted_100',
    'diff_distinct_CustomerID_grad_nonsorted_range',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_mean',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_std',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_0',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_10',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_20',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_30',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_40',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_50',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_60',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_70',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_80',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_90',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_100',
    'diff_distinct_CustomerID_ProductID_grad_nonsorted_range',


    'diff_distinct_ProductID_grad_nonsorted_mean',
    'diff_distinct_ProductID_grad_nonsorted_std',
    'diff_distinct_ProductID_grad_nonsorted_range',
    'diff_distinct_ProductID_grad_nonsorted_0',
    'diff_distinct_ProductID_grad_nonsorted_10',
    'diff_distinct_ProductID_grad_nonsorted_20',
    'diff_distinct_ProductID_grad_nonsorted_30',
    'diff_distinct_ProductID_grad_nonsorted_40',
    'diff_distinct_ProductID_grad_nonsorted_50',
    'diff_distinct_ProductID_grad_nonsorted_60',
    'diff_distinct_ProductID_grad_nonsorted_70',
    'diff_distinct_ProductID_grad_nonsorted_80',
    'diff_distinct_ProductID_grad_nonsorted_90',
    'diff_distinct_ProductID_grad_nonsorted_100',
    
    'diff_distinct_ProductID_grad_grad_mean',
    'diff_distinct_ProductID_grad_grad_std',
    'diff_distinct_ProductID_grad_grad_0',
    'diff_distinct_ProductID_grad_grad_10',
    'diff_distinct_ProductID_grad_grad_20',
    'diff_distinct_ProductID_grad_grad_30',
    'diff_distinct_ProductID_grad_grad_40',
    'diff_distinct_ProductID_grad_grad_50',
    'diff_distinct_ProductID_grad_grad_60',
    'diff_distinct_ProductID_grad_grad_70',
    'diff_distinct_ProductID_grad_grad_80',
    'diff_distinct_ProductID_grad_grad_90',
    'diff_distinct_ProductID_grad_grad_100',
    'diff_distinct_ProductID_grad_grad_range',
    'diff_distinct_CustomerID_grad_grad_mean',
    'diff_distinct_CustomerID_grad_grad_std',
    'diff_distinct_CustomerID_grad_grad_0',
    'diff_distinct_CustomerID_grad_grad_10',
    'diff_distinct_CustomerID_grad_grad_20',
    'diff_distinct_CustomerID_grad_grad_30',
    'diff_distinct_CustomerID_grad_grad_40',
    'diff_distinct_CustomerID_grad_grad_50',
    'diff_distinct_CustomerID_grad_grad_60',
    'diff_distinct_CustomerID_grad_grad_70',
    'diff_distinct_CustomerID_grad_grad_80',
    'diff_distinct_CustomerID_grad_grad_90',
    'diff_distinct_CustomerID_grad_grad_100',
    'diff_distinct_CustomerID_grad_grad_range',
    'diff_distinct_CustomerID_ProductID_grad_grad_mean',
    'diff_distinct_CustomerID_ProductID_grad_grad_std',
    'diff_distinct_CustomerID_ProductID_grad_grad_0',
    'diff_distinct_CustomerID_ProductID_grad_grad_10',
    'diff_distinct_CustomerID_ProductID_grad_grad_20',
    'diff_distinct_CustomerID_ProductID_grad_grad_30',
    'diff_distinct_CustomerID_ProductID_grad_grad_40',
    'diff_distinct_CustomerID_ProductID_grad_grad_50',
    'diff_distinct_CustomerID_ProductID_grad_grad_60',
    'diff_distinct_CustomerID_ProductID_grad_grad_70',
    'diff_distinct_CustomerID_ProductID_grad_grad_80',
    'diff_distinct_CustomerID_ProductID_grad_grad_90',
    'diff_distinct_CustomerID_ProductID_grad_grad_100',
    'diff_distinct_CustomerID_ProductID_grad_grad_range',
    
    
    'diff_distinct_high_virus_CustomerID_grad_mean_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_std_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_0_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_10_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_20_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_30_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_40_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_50_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_60_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_70_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_80_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_90_non_sorted',
    'diff_distinct_high_virus_CustomerID_grad_100_non_sorted',
    'diff_high_virus_CustomerID_grad_mean_non_sorted',
    'diff_high_virus_CustomerID_grad_std_non_sorted',
    'diff_high_virus_CustomerID_grad_0_non_sorted',
    'diff_high_virus_CustomerID_grad_10_non_sorted',
    'diff_high_virus_CustomerID_grad_20_non_sorted',
    'diff_high_virus_CustomerID_grad_30_non_sorted',
    'diff_high_virus_CustomerID_grad_40_non_sorted',
    'diff_high_virus_CustomerID_grad_50_non_sorted',
    'diff_high_virus_CustomerID_grad_60_non_sorted',
    'diff_high_virus_CustomerID_grad_70_non_sorted',
    'diff_high_virus_CustomerID_grad_80_non_sorted',
    'diff_high_virus_CustomerID_grad_90_non_sorted',
    'diff_high_virus_CustomerID_grad_100_non_sorted',
    'cv_hour_ratio_mean', 'cv_hour_ratio_std', 'cv_hour_ratio_range',
    'cv_hour_ratio_0', 'cv_hour_ratio_10', 'cv_hour_ratio_20', 'cv_hour_ratio_30', 'cv_hour_ratio_40', 'cv_hour_ratio_50', 'cv_hour_ratio_60', 'cv_hour_ratio_70', 'cv_hour_ratio_80', 'cv_hour_ratio_90', 'cv_hour_ratio_100',

    'QueryTS_10', ' QueryTS_20', ' QueryTS_30', ' QueryTS_40', ' QueryTS_60', ' QueryTS_80', ' QueryTS_90', 
        'QueryTS_mean', ' QueryTS_std', ' QueryTS_min', ' QueryTS_50', ' QueryTS_70', ' QueryTS_max',

        'by_hour_10', ' by_hour_20', ' by_hour_30', ' by_hour_40', ' by_hour_60', ' by_hour_80', ' by_hour_90',
        'by_hour_mean', ' by_hour_std', ' by_hour_min', ' by_hour_50', ' by_hour_70', ' by_hour_max', 

        'by_weekday_10', ' by_weekday_20', ' by_weekday_30', ' by_weekday_40', ' by_weekday_60', ' by_weekday_80', ' by_weekday_90',
        'by_weekday_mean', ' by_weekday_std', ' by_weekday_min', ' by_weekday_50', ' by_weekday_70', ' by_weekday_max',
  
    'ratio_virus_customer',
    'diff_distinct_high_virus_CustomerID_grad_mean', 'diff_distinct_high_virus_CustomerID_grad_std',

    'diff_distinct_high_virus_CustomerID_grad_0', 'diff_distinct_high_virus_CustomerID_grad_10', 'diff_distinct_high_virus_CustomerID_grad_20', 'diff_distinct_high_virus_CustomerID_grad_30', 'diff_distinct_high_virus_CustomerID_grad_40', 'diff_distinct_high_virus_CustomerID_grad_50', 'diff_distinct_high_virus_CustomerID_grad_60', 'diff_distinct_high_virus_CustomerID_grad_70', 'diff_distinct_high_virus_CustomerID_grad_80', 'diff_distinct_high_virus_CustomerID_grad_90', 'diff_distinct_high_virus_CustomerID_grad_100',

    'diff_distinct_high_virus_CustomerID_mean', 'diff_distinct_high_virus_CustomerID_0', 'diff_distinct_high_virus_CustomerID_10', 'diff_distinct_high_virus_CustomerID_20', 'diff_distinct_high_virus_CustomerID_30', 'diff_distinct_high_virus_CustomerID_40', 'diff_distinct_high_virus_CustomerID_50', 'diff_distinct_high_virus_CustomerID_60', 'diff_distinct_high_virus_CustomerID_70', 'diff_distinct_high_virus_CustomerID_80', 'diff_distinct_high_virus_CustomerID_90', 'diff_distinct_high_virus_CustomerID_100',

    'ratio_virus_submission',
    'diff_high_virus_CustomerID_grad_mean', 'diff_high_virus_CustomerID_grad_std',

    'diff_high_virus_CustomerID_grad_0', 'diff_high_virus_CustomerID_grad_10', 'diff_high_virus_CustomerID_grad_20', 'diff_high_virus_CustomerID_grad_30', 'diff_high_virus_CustomerID_grad_40', 'diff_high_virus_CustomerID_grad_50', 'diff_high_virus_CustomerID_grad_60', 'diff_high_virus_CustomerID_grad_70', 'diff_high_virus_CustomerID_grad_80', 'diff_high_virus_CustomerID_grad_90', 'diff_high_virus_CustomerID_grad_100',
    'diff_high_virus_CustomerID_mean', 'diff_high_virus_CustomerID_0', 'diff_high_virus_CustomerID_10', 'diff_high_virus_CustomerID_20', 'diff_high_virus_CustomerID_30', 'diff_high_virus_CustomerID_40', 'diff_high_virus_CustomerID_50', 'diff_high_virus_CustomerID_60', 'diff_high_virus_CustomerID_70', 'diff_high_virus_CustomerID_80', 'diff_high_virus_CustomerID_90', 'diff_high_virus_CustomerID_100',

    

    'customer_uniq_ratio_mean', 'customer_uniq_ratio_std', 'customer_uniq_ratio_range',
    'customer_uniq_ratio_0', 'customer_uniq_ratio_10', 'customer_uniq_ratio_20', 'customer_uniq_ratio_30', 'customer_uniq_ratio_40', 'customer_uniq_ratio_50', 'customer_uniq_ratio_60', 'customer_uniq_ratio_70', 'customer_uniq_ratio_80', 'customer_uniq_ratio_90', 'customer_uniq_ratio_100',
    'customer_product_ratio_mean', 'customer_product_ratio_std', 'customer_product_ratio_range',
    'customer_product_ratio_0', 'customer_product_ratio_10', 'customer_product_ratio_20', 'customer_product_ratio_30', 'customer_product_ratio_40', 'customer_product_ratio_50', 'customer_product_ratio_60', 'customer_product_ratio_70', 'customer_product_ratio_80', 'customer_product_ratio_90', 'customer_product_ratio_100',
    
    'product_ratio_mean', 'product_ratio_std', 'product_ratio_range',
    'product_ratio_0', 'product_ratio_10', 'product_ratio_20', 'product_ratio_30', 'product_ratio_40', 'product_ratio_50', 'product_ratio_60', 'product_ratio_70', 'product_ratio_80', 'product_ratio_90', 'product_ratio_100',
    

    'CustomerID_cluster_diff_g0_ratio',
    'CustomerID_cluster_diff_g1_ratio',
    'CustomerID_cluster_diff_g2_ratio',
    'CustomerID_cluster_diff_g3_ratio',
    'CustomerID_cluster_diff_g4_ratio',
    'CustomerID_cluster_diff_g5_ratio',
    'CustomerID_cluster_diff_gna_ratio',

    'CustomerID_cluster_diff_g0_count',
    'CustomerID_cluster_diff_g1_count',
    'CustomerID_cluster_diff_g2_count',
    'CustomerID_cluster_diff_g3_count',
    'CustomerID_cluster_diff_g4_count',
    'CustomerID_cluster_diff_g5_count',
    'CustomerID_cluster_diff_gna_count',

    'CustomerID_ProductID_cluster_diff_g0_ratio',
    'CustomerID_ProductID_cluster_diff_g1_ratio',
    'CustomerID_ProductID_cluster_diff_g2_ratio',
    'CustomerID_ProductID_cluster_diff_g3_ratio',
    'CustomerID_ProductID_cluster_diff_gna_ratio',

    'CustomerID_ProductID_cluster_diff_g0_count',
    'CustomerID_ProductID_cluster_diff_g1_count',
    'CustomerID_ProductID_cluster_diff_g2_count',
    'CustomerID_ProductID_cluster_diff_g3_count',
    'CustomerID_ProductID_cluster_diff_gna_count',
    
    'ratio_cc3a6a',
                'ratio_634e6b',
                'ratio_0374c4',
                'ratio_8541a0',
                'ratio_dd8d4a',
                'ratio_b93794',
                'ratio_a310bb',
                'ratio_3ea8c3',
                'ratio_8b7f69',
                'ratio_8452da',
                'ratio_26a5d0',
                'ratio_3c2be6',
                'ratio_055649',
                'ratio_20f8a5',
                'ratio_55649',
                'ratio_fec24f',
                'ratio_218578',
                'ratio_262880',
                'ratio_c76d58',
                'ratio_d465fc',
                'ratio_7acab3',
                'ratio_75f310',
                'ratio_c105a0',
                'ratio_e47f04',
                'ratio_533133',
                'ratio_05b409',
                'ratio_885fab',
                'ratio_aaa9c8',
                'ratio_0cdb7a',
                'count_cc3a6a',
                'count_634e6b',
                'count_0374c4',
                'count_8541a0',
                'count_dd8d4a',
                'count_b93794',
                'count_a310bb',
                'count_3ea8c3',
                'count_8b7f69',
                'count_8452da',
                'count_26a5d0',
                'count_3c2be6',
                'count_055649',
                'count_20f8a5',
                'count_55649',
                'count_fec24f',
                'count_218578',
                'count_262880',
                'count_c76d58',
                'count_d465fc',
                'count_7acab3',
                'count_75f310',
                'count_c105a0',
                'count_e47f04',
                'count_533133',
                'count_05b409',
                'count_885fab',
                'count_aaa9c8',
                'count_0cdb7a',
                'count_cc3a6a_uniq_CustomerID',
                'count_634e6b_uniq_CustomerID',
                'count_0374c4_uniq_CustomerID',
                'count_8541a0_uniq_CustomerID',
                'count_dd8d4a_uniq_CustomerID',
                'count_b93794_uniq_CustomerID',
                'count_a310bb_uniq_CustomerID',
                'count_3ea8c3_uniq_CustomerID',
                'count_8b7f69_uniq_CustomerID',
                'count_8452da_uniq_CustomerID',
                'count_26a5d0_uniq_CustomerID',
                'count_3c2be6_uniq_CustomerID',
                'count_055649_uniq_CustomerID',
                'count_20f8a5_uniq_CustomerID',
                'count_55649_uniq_CustomerID',
                'count_fec24f_uniq_CustomerID',
                'count_218578_uniq_CustomerID',
                'count_262880_uniq_CustomerID',
                'count_c76d58_uniq_CustomerID',
                'count_d465fc_uniq_CustomerID',
                'count_7acab3_uniq_CustomerID',
                'count_75f310_uniq_CustomerID',
                'count_c105a0_uniq_CustomerID',
                'count_e47f04_uniq_CustomerID',
                'count_533133_uniq_CustomerID',
                'count_05b409_uniq_CustomerID',
                'count_885fab_uniq_CustomerID',
                'count_aaa9c8_uniq_CustomerID',
                'count_0cdb7a_uniq_CustomerID',
                'ratio_cc3a6a_uniq_CustomerID',
                'ratio_634e6b_uniq_CustomerID',
                'ratio_0374c4_uniq_CustomerID',
                'ratio_8541a0_uniq_CustomerID',
                'ratio_dd8d4a_uniq_CustomerID',
                'ratio_b93794_uniq_CustomerID',
                'ratio_a310bb_uniq_CustomerID',
                'ratio_3ea8c3_uniq_CustomerID',
                'ratio_8b7f69_uniq_CustomerID',
                'ratio_8452da_uniq_CustomerID',
                'ratio_26a5d0_uniq_CustomerID',
                'ratio_3c2be6_uniq_CustomerID',
                'ratio_055649_uniq_CustomerID',
                'ratio_20f8a5_uniq_CustomerID',
                'ratio_55649_uniq_CustomerID',
                'ratio_fec24f_uniq_CustomerID',
                'ratio_218578_uniq_CustomerID',
                'ratio_262880_uniq_CustomerID',
                'ratio_c76d58_uniq_CustomerID',
                'ratio_d465fc_uniq_CustomerID',
                'ratio_7acab3_uniq_CustomerID',
                'ratio_75f310_uniq_CustomerID',
                'ratio_c105a0_uniq_CustomerID',
                'ratio_e47f04_uniq_CustomerID',
                'ratio_533133_uniq_CustomerID',
                'ratio_05b409_uniq_CustomerID',
                'ratio_885fab_uniq_CustomerID',
                'ratio_aaa9c8_uniq_CustomerID',
                'ratio_0cdb7a_uniq_CustomerID',
    'time_to_complete_10', 'time_to_complete_20', 'time_to_complete_30', 'time_to_complete_40', 'time_to_complete_50',
    'time_to_complete_60', 'time_to_complete_70', 'time_to_complete_80', 'time_to_complete_90', 'time_to_complete_10_uniq_customer', 
    'time_to_complete_20_uniq_customer', 'time_to_complete_30_uniq_customer', 'time_to_complete_40_uniq_customer', 'time_to_complete_50_uniq_customer', 'time_to_complete_60_uniq_customer', 'time_to_complete_70_uniq_customer', 'time_to_complete_80_uniq_customer', 'time_to_complete_90_uniq_customer', 'time_to_complete_10_uniq_product', 'time_to_complete_20_uniq_product', 'time_to_complete_30_uniq_product', 'time_to_complete_40_uniq_product', 'time_to_complete_50_uniq_product', 'time_to_complete_60_uniq_product', 'time_to_complete_70_uniq_product', 'time_to_complete_80_uniq_product', 'time_to_complete_90_uniq_product',
    
    
        'customer_ratio_mean', 'customer_ratio_std', 'customer_ratio_range',
        'customer_ratio_0', 'customer_ratio_10', 'customer_ratio_20', 'customer_ratio_30', 'customer_ratio_40', 'customer_ratio_50', 'customer_ratio_60', 'customer_ratio_70', 'customer_ratio_80', 'customer_ratio_90', 'customer_ratio_100',
        
        ##
        'diff_distinct_CustomerID_grad_mean', 'diff_distinct_CustomerID_grad_std',
        'diff_distinct_CustomerID_grad_0', 'diff_distinct_CustomerID_grad_10', 'diff_distinct_CustomerID_grad_20', 'diff_distinct_CustomerID_grad_30', 'diff_distinct_CustomerID_grad_40', 'diff_distinct_CustomerID_grad_50', 'diff_distinct_CustomerID_grad_60', 'diff_distinct_CustomerID_grad_70', 'diff_distinct_CustomerID_grad_80', 'diff_distinct_CustomerID_grad_90', 'diff_distinct_CustomerID_grad_100',
        ##

        'diff_distinct_CustomerID_ProductID_grad_mean',
        'diff_distinct_CustomerID_ProductID_grad_std',
        'diff_distinct_CustomerID_ProductID_grad_0', 'diff_distinct_CustomerID_ProductID_grad_10', 'diff_distinct_CustomerID_ProductID_grad_20', 'diff_distinct_CustomerID_ProductID_grad_30', 'diff_distinct_CustomerID_ProductID_grad_40', 'diff_distinct_CustomerID_ProductID_grad_50', 'diff_distinct_CustomerID_ProductID_grad_60', 'diff_distinct_CustomerID_ProductID_grad_70', 'diff_distinct_CustomerID_ProductID_grad_80', 'diff_distinct_CustomerID_ProductID_grad_90', 'diff_distinct_CustomerID_ProductID_grad_100',
        'diff_distinct_CustomerID_ProductID_grad_range',

        'diff_distinct_ProductID_range',
        'diff_distinct_ProductID_grad_range',
        'diff_distinct_ProductID_top_range',
        'diff_distinct_CustomerID_range',
        'diff_distinct_CustomerID_grad_range',
        'diff_distinct_CustomerID_top_range',

        'diff_distinct_ProductID_grad_0', 'diff_distinct_ProductID_grad_10', 'diff_distinct_ProductID_grad_20', 'diff_distinct_ProductID_grad_30', 'diff_distinct_ProductID_grad_40', 'diff_distinct_ProductID_grad_50', 'diff_distinct_ProductID_grad_60', 'diff_distinct_ProductID_grad_70', 'diff_distinct_ProductID_grad_80', 'diff_distinct_ProductID_grad_90', 'diff_distinct_ProductID_grad_100',
        #'diff_distinct_CustomerID_0', 'diff_distinct_CustomerID_10', 'diff_distinct_CustomerID_20', 'diff_distinct_CustomerID_30', 'diff_distinct_CustomerID_40', 'diff_distinct_CustomerID_50', 'diff_distinct_CustomerID_60', 'diff_distinct_CustomerID_70', 'diff_distinct_CustomerID_80', 'diff_distinct_CustomerID_90', 'diff_distinct_CustomerID_100',
        'seldom_ratio',
        'diff_distinct_ProductID_grad_mean', 'diff_distinct_ProductID_grad_std',

            'diff_distinct_ProductID_std', 
    #'diff_distinct_ProductID_top_std', 
    
    'diff_distinct_CustomerID_std', 'diff_distinct_CustomerID_top_std',
            'diff_distinct_ProductID_mean', 'diff_distinct_ProductID_0', 'diff_distinct_ProductID_10', 'diff_distinct_ProductID_20', 'diff_distinct_ProductID_30', 'diff_distinct_ProductID_40', 'diff_distinct_ProductID_50', 'diff_distinct_ProductID_60', 'diff_distinct_ProductID_70', 'diff_distinct_ProductID_80', 'diff_distinct_ProductID_90', 'diff_distinct_ProductID_100',
            'diff_distinct_ProductID_top_mean', 'diff_distinct_ProductID_top_0', 'diff_distinct_ProductID_top_10', 'diff_distinct_ProductID_top_20', 'diff_distinct_ProductID_top_30', 'diff_distinct_ProductID_top_40', 'diff_distinct_ProductID_top_50', 'diff_distinct_ProductID_top_60', 'diff_distinct_ProductID_top_70', 'diff_distinct_ProductID_top_80', 'diff_distinct_ProductID_top_90', 'diff_distinct_ProductID_top_100',
            'diff_distinct_CustomerID_mean', 'diff_distinct_CustomerID_0', 'diff_distinct_CustomerID_10', 'diff_distinct_CustomerID_20', 'diff_distinct_CustomerID_30', 'diff_distinct_CustomerID_40', 'diff_distinct_CustomerID_50', 'diff_distinct_CustomerID_60', 'diff_distinct_CustomerID_70', 'diff_distinct_CustomerID_80', 'diff_distinct_CustomerID_90', 'diff_distinct_CustomerID_100',
            'diff_distinct_CustomerID_top_mean', 'diff_distinct_CustomerID_top_0', 'diff_distinct_CustomerID_top_10', 'diff_distinct_CustomerID_top_20', 'diff_distinct_CustomerID_top_30', 'diff_distinct_CustomerID_top_40', 'diff_distinct_CustomerID_top_50', 'diff_distinct_CustomerID_top_60', 'diff_distinct_CustomerID_top_70', 'diff_distinct_CustomerID_top_80', 'diff_distinct_CustomerID_top_90', 'diff_distinct_CustomerID_top_100',
            
        '_duration_95_05',
        '_duration_80_20',
        'hour_ratio_1',
         'hour_ratio_2',
         'hour_ratio_3',
         'hour_ratio_4',
         'hour_ratio_5',
         'hour_ratio_6',
         'hour_ratio_7',
         'hour_ratio_8',
         'hour_ratio_9',
         'hour_ratio_10',
         'hour_ratio_11',
         'hour_ratio_12',
         'hour_ratio_13',
         'hour_ratio_14',
         'hour_ratio_15',
         'hour_ratio_16',
         'hour_ratio_17',
         'hour_ratio_18',
         'hour_ratio_19',
         'hour_ratio_20',
         'hour_ratio_21',
         'hour_ratio_22',
         'hour_ratio_23',
         'hour_ratio_24',
        'weekday_ratio_0','weekday_ratio_1','weekday_ratio_2','weekday_ratio_3','weekday_ratio_4','weekday_ratio_5','weekday_ratio_6',\
    'diff_10', 'diff_20', 'diff_30', 'diff_40', 'diff_60', 'diff_80', 'diff_90', \
        'diff_by_CustomerID_10', 'diff_by_CustomerID_20', 'diff_by_CustomerID_30', 'diff_by_CustomerID_40', 'diff_by_CustomerID_60', 'diff_by_CustomerID_80', 'diff_by_CustomerID_90',\
        'diff_by_CustomerID_count', ' diff_by_CustomerID_mean', ' diff_by_CustomerID_std', ' diff_by_CustomerID_min', ' diff_by_CustomerID_25', ' diff_by_CustomerID_50', ' diff_by_CustomerID_70', ' diff_by_CustomerID_max',\
        'diff_by_ProductID_10', 'diff_by_ProductID_20', 'diff_by_ProductID_30', 'diff_by_ProductID_40', 'diff_by_ProductID_60', 'diff_by_ProductID_80', 'diff_by_ProductID_90',\
        'diff_by_ProductID_count', ' diff_by_ProductID_mean', ' diff_by_ProductID_std', ' diff_by_ProductID_min', ' diff_by_ProductID_25', ' diff_by_ProductID_50', ' diff_by_ProductID_70', ' diff_by_ProductID_max',\
    'uniq_nb_ProductID_g0_ratio_3_by_uniq', 'uniq_nb_ProductID_g1_ratio_3_by_uniq', 'uniq_nb_ProductID_g2_ratio_3_by_uniq',\
        'uniq_nb_ratio_ProductID_g0_ratio_3_by_uniq', 'uniq_nb_ratio_ProductID_g1_ratio_3_by_uniq', 'uniq_nb_ratio_ProductID_g2_ratio_3_by_uniq',\
        'uniq_nb_CustomerID_g0_ratio_4_by_uniq', 'uniq_nb_CustomerID_g1_ratio_4_by_uniq', 'uniq_nb_CustomerID_g2_ratio_4_by_uniq', 'uniq_nb_CustomerID_g3_ratio_4_by_uniq',\
        'uniq_nb_ratio_CustomerID_g0_ratio_4_by_uniq', 'uniq_nb_ratio_CustomerID_g1_ratio_4_by_uniq', 
    'uniq_nb_ratio_CustomerID_g3_ratio_4_by_uniq',\
    '_ProductID_g0_ratio_2_by_uniq', '_ProductID_g1_ratio_2_by_uniq',\
        '_ProductID_g0_ratio_3_by_uniq', '_ProductID_g1_ratio_3_by_uniq', '_ProductID_g2_ratio_3_by_uniq',\
        '_CustomerID_g0_ratio_4_by_uniq', '_CustomerID_g1_ratio_4_by_uniq', '_CustomerID_g2_ratio_4_by_uniq', '_CustomerID_g3_ratio_4_by_uniq',\
        '_CustomerID_g0_ratio_5_by_usetime', '_CustomerID_g1_ratio_5_by_usetime', '_CustomerID_g2_ratio_5_by_usetime',\
        '_CustomerID_g3_ratio_5_by_usetime', '_CustomerID_g4_ratio_5_by_usetime', '_CustomerID_g5_ratio_5_by_usetime', '_CustomerID_g6_ratio_5_by_usetime',\
    '_duration_90_10','_CustomerID_g0_ratio_5_by_time','_CustomerID_g1_ratio_5_by_time','_CustomerID_g2_ratio_5_by_time',\
            '_CustomerID_g3_ratio_5_by_time','_CustomerID_g4_ratio_5_by_time','_PrdocutID_g0_ratio_9', '_PrdocutID_g1_ratio_9', '_PrdocutID_g2_ratio_9',\
             '_PrdocutID_g3_ratio_9','_PrdocutID_g4_ratio_9', '_PrdocutID_g5_ratio_9', '_PrdocutID_g6_ratio_9',\
             '_PrdocutID_g7_ratio_9','_PrdocutID_g8_ratio_9','_CustomerID_g0_ratio_3','_CustomerID_g1_ratio_3',\
             '_CustomerID_g2_ratio_3','CustomerID_ProductID_QueryTS_gt1_count','CustomerID_ProductID_QueryTS_gt1_mean','CustomerID_ProductID_QueryTS_gt1_std','CustomerID_ProductID_QueryTS_gt1_min','CustomerID_ProductID_QueryTS_gt1_25',\
                    'CustomerID_ProductID_QueryTS_gt1_50','CustomerID_ProductID_QueryTS_gt1_70','CustomerID_ProductID_QueryTS_gt1_max','CustomerID_ProductID_QueryTS_gt1_count_gt_up','CustomerID_ProductID_QueryTS_gt1_count_lt_down',\
    '_PrdocutID_g0_ratio', '_PrdocutID_g1_ratio', '_PrdocutID_g2_ratio', '_PrdocutID_g3_ratio','diff_count', 'diff_mean', 'diff_std', 'diff_min', 'diff_25', 'diff_50', 'diff_70', 'diff_max','counts', '_22_01_ratio', '_04_15_ratio', '_weekday_0_ratio', '_weekday_235_ratio','CustomerID_count','CustomerID_mean','CustomerID_std','CustomerID_min','CustomerID_25',\
                    'CustomerID_50','CustomerID_70','CustomerID_max','CustomerID_count_gt_up','CustomerID_count_lt_down','CustomerID_count_gt_one',\
                    'ProductID_count','ProductID_mean','ProductID_std','ProductID_min','ProductID_25',\
                    'ProductID_50','ProductID_70','ProductID_max','ProductID_count_gt_up','ProductID_count_lt_down','ProductID_count_gt_one',\
                    'CustomerID_ProductID_count','CustomerID_ProductID_mean','CustomerID_ProductID_std','CustomerID_ProductID_min','CustomerID_ProductID_25',\
                    'CustomerID_ProductID_50','CustomerID_ProductID_70','CustomerID_ProductID_max','CustomerID_ProductID_count_gt_up','CustomerID_ProductID_count_lt_down','CustomerID_ProductID_count_gt_one',\
                    'CustomerID_ProductID_QueryTS_count','CustomerID_ProductID_QueryTS_mean','CustomerID_ProductID_QueryTS_std','CustomerID_ProductID_QueryTS_min','CustomerID_ProductID_QueryTS_25',\
                    'CustomerID_ProductID_QueryTS_50','CustomerID_ProductID_QueryTS_70','CustomerID_ProductID_QueryTS_max','CustomerID_ProductID_QueryTS_count_gt_up','CustomerID_ProductID_QueryTS_count_lt_down','CustomerID_ProductID_QueryTS_count_gt_one',\

                    'CustomerID_ProductID_QueryDateTime_date_count','CustomerID_ProductID_QueryDateTime_date_mean','CustomerID_ProductID_QueryDateTime_date_std','CustomerID_ProductID_QueryDateTime_date_min','CustomerID_ProductID_QueryDateTime_date_25',\
                    'CustomerID_ProductID_QueryDateTime_date_50','CustomerID_ProductID_QueryDateTime_date_70','CustomerID_ProductID_QueryDateTime_date_max','CustomerID_ProductID_QueryDateTime_date_count_gt_up','CustomerID_ProductID_QueryDateTime_date_count_lt_down','CustomerID_ProductID_QueryDateTime_date_count_gt_one',\

                    'CustomerID_ProductID_QueryDateTime_date_hour_count','CustomerID_ProductID_QueryDateTime_date_hour_mean','CustomerID_ProductID_QueryDateTime_date_hour_std','CustomerID_ProductID_QueryDateTime_date_hour_min','CustomerID_ProductID_QueryDateTime_date_hour_25',\
                    'CustomerID_ProductID_QueryDateTime_date_hour_50','CustomerID_ProductID_QueryDateTime_date_hour_70','CustomerID_ProductID_QueryDateTime_date_hour_max','CustomerID_ProductID_QueryDateTime_date_hour_count_gt_up','CustomerID_ProductID_QueryDateTime_date_hour_count_lt_down','CustomerID_ProductID_QueryDateTime_date_hour_count_gt_one',\

                    'CustomerID_ProductID_diff_count','CustomerID_ProductID_diff_mean','CustomerID_ProductID_diff_std','CustomerID_ProductID_diff_min','CustomerID_ProductID_diff_25',\
                    'CustomerID_ProductID_diff_50','CustomerID_ProductID_diff_70','CustomerID_ProductID_diff_max','CustomerID_ProductID_diff_count_gt_up','CustomerID_ProductID_diff_count_lt_down','CustomerID_ProductID_diff_count_gt_one',\
                    ##
                    'CustomerID_ProductID_QueryTS_one_file_count','CustomerID_ProductID_QueryTS_one_file_mean','CustomerID_ProductID_QueryTS_one_file_std','CustomerID_ProductID_QueryTS_one_file_min','CustomerID_ProductID_QueryTS_one_file_25',\
                    'CustomerID_ProductID_QueryTS_one_file_50','CustomerID_ProductID_QueryTS_one_file_70','CustomerID_ProductID_QueryTS_one_file_max','CustomerID_ProductID_QueryTS_one_file_count_gt_up','CustomerID_ProductID_QueryTS_one_file_count_lt_down','CustomerID_ProductID_QueryTS_one_file_count_gt_one',\

                    'CustomerID_ProductID_QueryDateTime_date_one_file_count','CustomerID_ProductID_QueryDateTime_date_one_file_mean','CustomerID_ProductID_QueryDateTime_date_one_file_std','CustomerID_ProductID_QueryDateTime_date_one_file_min','CustomerID_ProductID_QueryDateTime_date_one_file_25',\
                    'CustomerID_ProductID_QueryDateTime_date_one_file_50','CustomerID_ProductID_QueryDateTime_date_one_file_70','CustomerID_ProductID_QueryDateTime_date_one_file_max','CustomerID_ProductID_QueryDateTime_date_one_file_count_gt_up','CustomerID_ProductID_QueryDateTime_date_one_file_count_lt_down','CustomerID_ProductID_QueryDateTime_date_one_file_count_gt_one',\
                    'CustomerID_ProductID_QueryDateTime_date_hour_one_file_count','CustomerID_ProductID_QueryDateTime_date_hour_one_file_mean','CustomerID_ProductID_QueryDateTime_date_hour_one_file_std','CustomerID_ProductID_QueryDateTime_date_hour_one_file_min','CustomerID_ProductID_QueryDateTime_date_hour_one_file_25',\
                    'CustomerID_ProductID_QueryDateTime_date_hour_one_file_50','CustomerID_ProductID_QueryDateTime_date_hour_one_file_70','CustomerID_ProductID_QueryDateTime_date_hour_one_file_max','CustomerID_ProductID_QueryDateTime_date_hour_one_file_count_gt_up','CustomerID_ProductID_QueryDateTime_date_hour_one_file_count_lt_down','CustomerID_ProductID_QueryDateTime_date_hour_one_file_count_gt_one',\
                    'CustomerID_ProductID_diff_one_file_count','CustomerID_ProductID_diff_one_file_mean','CustomerID_ProductID_diff_one_file_std','CustomerID_ProductID_diff_one_file_min','CustomerID_ProductID_diff_one_file_25',\
                    'CustomerID_ProductID_diff_one_file_50','CustomerID_ProductID_diff_one_file_70','CustomerID_ProductID_diff_one_file_max','CustomerID_ProductID_diff_one_file_count_gt_up','CustomerID_ProductID_diff_one_file_count_lt_down','CustomerID_ProductID_diff_one_file_count_gt_one',\
                '_duration', '_3_month_ratio', '_4_month_ratio', '_5_month_ratio', '_6_month_ratio',
    # ae
    'ae_0', 'ae_1', 'ae_2', 'ae_3',
    'ae_all_0', 'ae_all_1', 'ae_all_2', 'ae_all_3', 'ae_all_4', 'ae_all_5',
       'ae_all_6', 'ae_all_7', 'ae_all_8', 'ae_all_9',
    'ae_all_0322_epochs1000_0',
 'ae_all_0322_epochs1000_1',
 'ae_all_0322_epochs1000_2',
 'ae_all_0322_epochs1000_3',
 'ae_all_0322_epochs1000_4',
 'ae_all_0322_epochs1000_5',
 'ae_all_0322_epochs1000_6',
 'ae_all_0322_epochs1000_7',
 'ae_all_0322_epochs1000_8',
 'ae_all_0322_epochs1000_9',
 'ae_all_0322_epochs1000_10',
 'ae_all_0322_epochs1000_11',
 'ae_all_0322_epochs1000_12',
      ]

# test data 表現最好的feature
var_best = [
   'cv_hour_ratio_mean', 'cv_hour_ratio_std', 'cv_hour_ratio_range',
    'cv_hour_ratio_0', 'cv_hour_ratio_10', 'cv_hour_ratio_20', 'cv_hour_ratio_30', 'cv_hour_ratio_40', 'cv_hour_ratio_50', 'cv_hour_ratio_60', 'cv_hour_ratio_70', 'cv_hour_ratio_80', 'cv_hour_ratio_90', 'cv_hour_ratio_100',
    'QueryTS_10', ' QueryTS_20', ' QueryTS_30', ' QueryTS_40', ' QueryTS_60', ' QueryTS_80', ' QueryTS_90', 
        'QueryTS_mean', ' QueryTS_std', ' QueryTS_min', ' QueryTS_50', ' QueryTS_70', ' QueryTS_max',
        'by_hour_10', ' by_hour_20', ' by_hour_30', ' by_hour_40', ' by_hour_60', ' by_hour_80', ' by_hour_90',
        'by_hour_mean', ' by_hour_std', ' by_hour_min', ' by_hour_50', ' by_hour_70', ' by_hour_max', 
        'by_weekday_10', ' by_weekday_20', ' by_weekday_30', ' by_weekday_40', ' by_weekday_60', ' by_weekday_80', ' by_weekday_90',
        'by_weekday_mean', ' by_weekday_std', ' by_weekday_min', ' by_weekday_50', ' by_weekday_70', ' by_weekday_max',
    'ratio_virus_customer',
    'diff_distinct_high_virus_CustomerID_grad_mean', 'diff_distinct_high_virus_CustomerID_grad_std',
    'diff_distinct_high_virus_CustomerID_grad_0', 'diff_distinct_high_virus_CustomerID_grad_10', 'diff_distinct_high_virus_CustomerID_grad_20', 'diff_distinct_high_virus_CustomerID_grad_30', 'diff_distinct_high_virus_CustomerID_grad_40', 'diff_distinct_high_virus_CustomerID_grad_50', 'diff_distinct_high_virus_CustomerID_grad_60', 'diff_distinct_high_virus_CustomerID_grad_70', 'diff_distinct_high_virus_CustomerID_grad_80', 'diff_distinct_high_virus_CustomerID_grad_90', 'diff_distinct_high_virus_CustomerID_grad_100',
    'diff_distinct_high_virus_CustomerID_mean', 'diff_distinct_high_virus_CustomerID_0', 'diff_distinct_high_virus_CustomerID_10', 'diff_distinct_high_virus_CustomerID_20', 'diff_distinct_high_virus_CustomerID_30', 'diff_distinct_high_virus_CustomerID_40', 'diff_distinct_high_virus_CustomerID_50', 'diff_distinct_high_virus_CustomerID_60', 'diff_distinct_high_virus_CustomerID_70', 'diff_distinct_high_virus_CustomerID_80', 'diff_distinct_high_virus_CustomerID_90', 'diff_distinct_high_virus_CustomerID_100',
    'ratio_virus_submission',
    'diff_high_virus_CustomerID_grad_mean', 'diff_high_virus_CustomerID_grad_std',
    'diff_high_virus_CustomerID_grad_0', 'diff_high_virus_CustomerID_grad_10', 'diff_high_virus_CustomerID_grad_20', 'diff_high_virus_CustomerID_grad_30', 'diff_high_virus_CustomerID_grad_40', 'diff_high_virus_CustomerID_grad_50', 'diff_high_virus_CustomerID_grad_60', 'diff_high_virus_CustomerID_grad_70', 'diff_high_virus_CustomerID_grad_80', 'diff_high_virus_CustomerID_grad_90', 'diff_high_virus_CustomerID_grad_100',
    'diff_high_virus_CustomerID_mean', 'diff_high_virus_CustomerID_0', 'diff_high_virus_CustomerID_10', 'diff_high_virus_CustomerID_20', 'diff_high_virus_CustomerID_30', 'diff_high_virus_CustomerID_40', 'diff_high_virus_CustomerID_50', 'diff_high_virus_CustomerID_60', 'diff_high_virus_CustomerID_70', 'diff_high_virus_CustomerID_80', 'diff_high_virus_CustomerID_90', 'diff_high_virus_CustomerID_100',
    'customer_uniq_ratio_mean', 'customer_uniq_ratio_std', 'customer_uniq_ratio_range',
    'customer_uniq_ratio_0', 'customer_uniq_ratio_10', 'customer_uniq_ratio_20', 'customer_uniq_ratio_30', 'customer_uniq_ratio_40', 'customer_uniq_ratio_50', 'customer_uniq_ratio_60', 'customer_uniq_ratio_70', 'customer_uniq_ratio_80', 'customer_uniq_ratio_90', 'customer_uniq_ratio_100',
    'customer_product_ratio_mean', 'customer_product_ratio_std', 'customer_product_ratio_range',
    'customer_product_ratio_0', 'customer_product_ratio_10', 'customer_product_ratio_20', 'customer_product_ratio_30', 'customer_product_ratio_40', 'customer_product_ratio_50', 'customer_product_ratio_60', 'customer_product_ratio_70', 'customer_product_ratio_80', 'customer_product_ratio_90', 'customer_product_ratio_100',
    'product_ratio_mean', 'product_ratio_std', 'product_ratio_range',
    'product_ratio_0', 'product_ratio_10', 'product_ratio_20', 'product_ratio_30', 'product_ratio_40', 'product_ratio_50', 'product_ratio_60', 'product_ratio_70', 'product_ratio_80', 'product_ratio_90', 'product_ratio_100',
    'CustomerID_cluster_diff_g0_ratio',
    'CustomerID_cluster_diff_g1_ratio',
    'CustomerID_cluster_diff_g2_ratio',
    'CustomerID_cluster_diff_g3_ratio',
    'CustomerID_cluster_diff_g4_ratio',
    'CustomerID_cluster_diff_g5_ratio',
    'CustomerID_cluster_diff_gna_ratio',
    'CustomerID_cluster_diff_g0_count',
    'CustomerID_cluster_diff_g1_count',
    'CustomerID_cluster_diff_g2_count',
    'CustomerID_cluster_diff_g3_count',
    'CustomerID_cluster_diff_g4_count',
    'CustomerID_cluster_diff_g5_count',
    'CustomerID_cluster_diff_gna_count',
    'CustomerID_ProductID_cluster_diff_g0_ratio',
    'CustomerID_ProductID_cluster_diff_g1_ratio',
    'CustomerID_ProductID_cluster_diff_g2_ratio',
    'CustomerID_ProductID_cluster_diff_g3_ratio',
    'CustomerID_ProductID_cluster_diff_gna_ratio',
    'CustomerID_ProductID_cluster_diff_g0_count',
    'CustomerID_ProductID_cluster_diff_g1_count',
    'CustomerID_ProductID_cluster_diff_g2_count',
    'CustomerID_ProductID_cluster_diff_g3_count',
    'CustomerID_ProductID_cluster_diff_gna_count',
    'ratio_cc3a6a',
                'ratio_634e6b',
                'ratio_0374c4',
                'ratio_8541a0',
                'ratio_dd8d4a',
                'ratio_b93794',
                'ratio_a310bb',
                'ratio_3ea8c3',
                'ratio_8b7f69',
                'ratio_8452da',
                'ratio_26a5d0',
                'ratio_3c2be6',
                'ratio_055649',
                'ratio_20f8a5',
                'ratio_55649',
                'ratio_fec24f',
                'ratio_218578',
                'ratio_262880',
                'ratio_c76d58',
                'ratio_d465fc',
                'ratio_7acab3',
                'ratio_75f310',
                'ratio_c105a0',
                'ratio_e47f04',
                'ratio_533133',
                'ratio_05b409',
                'ratio_885fab',
                'ratio_aaa9c8',
                'ratio_0cdb7a',
                'count_cc3a6a',
                'count_634e6b',
                'count_0374c4',
                'count_8541a0',
                'count_dd8d4a',
                'count_b93794',
                'count_a310bb',
                'count_3ea8c3',
                'count_8b7f69',
                'count_8452da',
                'count_26a5d0',
                'count_3c2be6',
                'count_055649',
                'count_20f8a5',
                'count_55649',
                'count_fec24f',
                'count_218578',
                'count_262880',
                'count_c76d58',
                'count_d465fc',
                'count_7acab3',
                'count_75f310',
                'count_c105a0',
                'count_e47f04',
                'count_533133',
                'count_05b409',
                'count_885fab',
                'count_aaa9c8',
                'count_0cdb7a',
                'count_cc3a6a_uniq_CustomerID',
                'count_634e6b_uniq_CustomerID',
                'count_0374c4_uniq_CustomerID',
                'count_8541a0_uniq_CustomerID',
                'count_dd8d4a_uniq_CustomerID',
                'count_b93794_uniq_CustomerID',
                'count_a310bb_uniq_CustomerID',
                'count_3ea8c3_uniq_CustomerID',
                'count_8b7f69_uniq_CustomerID',
                'count_8452da_uniq_CustomerID',
                'count_26a5d0_uniq_CustomerID',
                'count_3c2be6_uniq_CustomerID',
                'count_055649_uniq_CustomerID',
                'count_20f8a5_uniq_CustomerID',
                'count_55649_uniq_CustomerID',
                'count_fec24f_uniq_CustomerID',
                'count_218578_uniq_CustomerID',
                'count_262880_uniq_CustomerID',
                'count_c76d58_uniq_CustomerID',
                'count_d465fc_uniq_CustomerID',
                'count_7acab3_uniq_CustomerID',
                'count_75f310_uniq_CustomerID',
                'count_c105a0_uniq_CustomerID',
                'count_e47f04_uniq_CustomerID',
                'count_533133_uniq_CustomerID',
                'count_05b409_uniq_CustomerID',
                'count_885fab_uniq_CustomerID',
                'count_aaa9c8_uniq_CustomerID',
                'count_0cdb7a_uniq_CustomerID',
                'ratio_cc3a6a_uniq_CustomerID',
                'ratio_634e6b_uniq_CustomerID',
                'ratio_0374c4_uniq_CustomerID',
                'ratio_8541a0_uniq_CustomerID',
                'ratio_dd8d4a_uniq_CustomerID',
                'ratio_b93794_uniq_CustomerID',
                'ratio_a310bb_uniq_CustomerID',
                'ratio_3ea8c3_uniq_CustomerID',
                'ratio_8b7f69_uniq_CustomerID',
                'ratio_8452da_uniq_CustomerID',
                'ratio_26a5d0_uniq_CustomerID',
                'ratio_3c2be6_uniq_CustomerID',
                'ratio_055649_uniq_CustomerID',
                'ratio_20f8a5_uniq_CustomerID',
                'ratio_55649_uniq_CustomerID',
                'ratio_fec24f_uniq_CustomerID',
                'ratio_218578_uniq_CustomerID',
                'ratio_262880_uniq_CustomerID',
                'ratio_c76d58_uniq_CustomerID',
                'ratio_d465fc_uniq_CustomerID',
                'ratio_7acab3_uniq_CustomerID',
                'ratio_75f310_uniq_CustomerID',
                'ratio_c105a0_uniq_CustomerID',
                'ratio_e47f04_uniq_CustomerID',
                'ratio_533133_uniq_CustomerID',
                'ratio_05b409_uniq_CustomerID',
                'ratio_885fab_uniq_CustomerID',
                'ratio_aaa9c8_uniq_CustomerID',
                'ratio_0cdb7a_uniq_CustomerID',
    'time_to_complete_10', 'time_to_complete_20', 'time_to_complete_30', 'time_to_complete_40', 'time_to_complete_50',
    'time_to_complete_60', 'time_to_complete_70', 'time_to_complete_80', 'time_to_complete_90', 'time_to_complete_10_uniq_customer', 
    'time_to_complete_20_uniq_customer', 'time_to_complete_30_uniq_customer', 'time_to_complete_40_uniq_customer', 'time_to_complete_50_uniq_customer', 'time_to_complete_60_uniq_customer', 'time_to_complete_70_uniq_customer', 'time_to_complete_80_uniq_customer', 'time_to_complete_90_uniq_customer', 'time_to_complete_10_uniq_product', 'time_to_complete_20_uniq_product', 'time_to_complete_30_uniq_product', 'time_to_complete_40_uniq_product', 'time_to_complete_50_uniq_product', 'time_to_complete_60_uniq_product', 'time_to_complete_70_uniq_product', 'time_to_complete_80_uniq_product', 'time_to_complete_90_uniq_product',
        'customer_ratio_mean', 'customer_ratio_std', 'customer_ratio_range',
        'customer_ratio_0', 'customer_ratio_10', 'customer_ratio_20', 'customer_ratio_30', 'customer_ratio_40', 'customer_ratio_50', 'customer_ratio_60', 'customer_ratio_70', 'customer_ratio_80', 'customer_ratio_90', 'customer_ratio_100',
        'diff_distinct_CustomerID_grad_mean', 'diff_distinct_CustomerID_grad_std',
        'diff_distinct_CustomerID_grad_0', 'diff_distinct_CustomerID_grad_10', 'diff_distinct_CustomerID_grad_20', 'diff_distinct_CustomerID_grad_30', 'diff_distinct_CustomerID_grad_40', 'diff_distinct_CustomerID_grad_50', 'diff_distinct_CustomerID_grad_60', 'diff_distinct_CustomerID_grad_70', 'diff_distinct_CustomerID_grad_80', 'diff_distinct_CustomerID_grad_90', 'diff_distinct_CustomerID_grad_100',
        'diff_distinct_CustomerID_ProductID_grad_mean',
        'diff_distinct_CustomerID_ProductID_grad_std',
        'diff_distinct_CustomerID_ProductID_grad_0', 'diff_distinct_CustomerID_ProductID_grad_10', 'diff_distinct_CustomerID_ProductID_grad_20', 'diff_distinct_CustomerID_ProductID_grad_30', 'diff_distinct_CustomerID_ProductID_grad_40', 'diff_distinct_CustomerID_ProductID_grad_50', 'diff_distinct_CustomerID_ProductID_grad_60', 'diff_distinct_CustomerID_ProductID_grad_70', 'diff_distinct_CustomerID_ProductID_grad_80', 'diff_distinct_CustomerID_ProductID_grad_90', 'diff_distinct_CustomerID_ProductID_grad_100',
        'diff_distinct_CustomerID_ProductID_grad_range',
        'diff_distinct_ProductID_range',
        'diff_distinct_ProductID_grad_range',
        'diff_distinct_ProductID_top_range',
        'diff_distinct_CustomerID_range',
        'diff_distinct_CustomerID_grad_range',
        'diff_distinct_CustomerID_top_range',
        'diff_distinct_ProductID_grad_0', 'diff_distinct_ProductID_grad_10', 'diff_distinct_ProductID_grad_20', 'diff_distinct_ProductID_grad_30', 'diff_distinct_ProductID_grad_40', 'diff_distinct_ProductID_grad_50', 'diff_distinct_ProductID_grad_60', 'diff_distinct_ProductID_grad_70', 'diff_distinct_ProductID_grad_80', 'diff_distinct_ProductID_grad_90', 'diff_distinct_ProductID_grad_100',
        'seldom_ratio',
        'diff_distinct_ProductID_grad_mean', 'diff_distinct_ProductID_grad_std',
            'diff_distinct_ProductID_std',     
    'diff_distinct_CustomerID_std', 'diff_distinct_CustomerID_top_std',
            'diff_distinct_ProductID_mean', 'diff_distinct_ProductID_0', 'diff_distinct_ProductID_10', 'diff_distinct_ProductID_20', 'diff_distinct_ProductID_30', 'diff_distinct_ProductID_40', 'diff_distinct_ProductID_50', 'diff_distinct_ProductID_60', 'diff_distinct_ProductID_70', 'diff_distinct_ProductID_80', 'diff_distinct_ProductID_90', 'diff_distinct_ProductID_100',
            'diff_distinct_ProductID_top_mean', 'diff_distinct_ProductID_top_0', 'diff_distinct_ProductID_top_10', 'diff_distinct_ProductID_top_20', 'diff_distinct_ProductID_top_30', 'diff_distinct_ProductID_top_40', 'diff_distinct_ProductID_top_50', 'diff_distinct_ProductID_top_60', 'diff_distinct_ProductID_top_70', 'diff_distinct_ProductID_top_80', 'diff_distinct_ProductID_top_90', 'diff_distinct_ProductID_top_100',
            'diff_distinct_CustomerID_mean', 'diff_distinct_CustomerID_0', 'diff_distinct_CustomerID_10', 'diff_distinct_CustomerID_20', 'diff_distinct_CustomerID_30', 'diff_distinct_CustomerID_40', 'diff_distinct_CustomerID_50', 'diff_distinct_CustomerID_60', 'diff_distinct_CustomerID_70', 'diff_distinct_CustomerID_80', 'diff_distinct_CustomerID_90', 'diff_distinct_CustomerID_100',
            'diff_distinct_CustomerID_top_mean', 'diff_distinct_CustomerID_top_0', 'diff_distinct_CustomerID_top_10', 'diff_distinct_CustomerID_top_20', 'diff_distinct_CustomerID_top_30', 'diff_distinct_CustomerID_top_40', 'diff_distinct_CustomerID_top_50', 'diff_distinct_CustomerID_top_60', 'diff_distinct_CustomerID_top_70', 'diff_distinct_CustomerID_top_80', 'diff_distinct_CustomerID_top_90', 'diff_distinct_CustomerID_top_100',
        '_duration_95_05',
        '_duration_80_20',
        'hour_ratio_1',
         'hour_ratio_2',
         'hour_ratio_3',
         'hour_ratio_4',
         'hour_ratio_5',
         'hour_ratio_6',
         'hour_ratio_7',
         'hour_ratio_8',
         'hour_ratio_9',
         'hour_ratio_10',
         'hour_ratio_11',
         'hour_ratio_12',
         'hour_ratio_13',
         'hour_ratio_14',
         'hour_ratio_15',
         'hour_ratio_16',
         'hour_ratio_17',
         'hour_ratio_18',
         'hour_ratio_19',
         'hour_ratio_20',
         'hour_ratio_21',
         'hour_ratio_22',
         'hour_ratio_23',
         'hour_ratio_24',
        'weekday_ratio_0','weekday_ratio_1','weekday_ratio_2','weekday_ratio_3','weekday_ratio_4','weekday_ratio_5','weekday_ratio_6',\
    'diff_10', 'diff_20', 'diff_30', 'diff_40', 'diff_60', 'diff_80', 'diff_90', \
        'diff_by_CustomerID_10', 'diff_by_CustomerID_20', 'diff_by_CustomerID_30', 'diff_by_CustomerID_40', 'diff_by_CustomerID_60', 'diff_by_CustomerID_80', 'diff_by_CustomerID_90',\
        'diff_by_CustomerID_count', ' diff_by_CustomerID_mean', ' diff_by_CustomerID_std', ' diff_by_CustomerID_min', ' diff_by_CustomerID_25', ' diff_by_CustomerID_50', ' diff_by_CustomerID_70', ' diff_by_CustomerID_max',\
        'diff_by_ProductID_10', 'diff_by_ProductID_20', 'diff_by_ProductID_30', 'diff_by_ProductID_40', 'diff_by_ProductID_60', 'diff_by_ProductID_80', 'diff_by_ProductID_90',\
        'diff_by_ProductID_count', ' diff_by_ProductID_mean', ' diff_by_ProductID_std', ' diff_by_ProductID_min', ' diff_by_ProductID_25', ' diff_by_ProductID_50', ' diff_by_ProductID_70', ' diff_by_ProductID_max',\
    'uniq_nb_ProductID_g0_ratio_3_by_uniq', 'uniq_nb_ProductID_g1_ratio_3_by_uniq', 'uniq_nb_ProductID_g2_ratio_3_by_uniq',\
        'uniq_nb_ratio_ProductID_g0_ratio_3_by_uniq', 'uniq_nb_ratio_ProductID_g1_ratio_3_by_uniq', 'uniq_nb_ratio_ProductID_g2_ratio_3_by_uniq',\
        'uniq_nb_CustomerID_g0_ratio_4_by_uniq', 'uniq_nb_CustomerID_g1_ratio_4_by_uniq', 'uniq_nb_CustomerID_g2_ratio_4_by_uniq', 'uniq_nb_CustomerID_g3_ratio_4_by_uniq',\
        'uniq_nb_ratio_CustomerID_g0_ratio_4_by_uniq', 'uniq_nb_ratio_CustomerID_g1_ratio_4_by_uniq', 
    'uniq_nb_ratio_CustomerID_g3_ratio_4_by_uniq',\
    '_ProductID_g0_ratio_2_by_uniq', '_ProductID_g1_ratio_2_by_uniq',\
        '_ProductID_g0_ratio_3_by_uniq', '_ProductID_g1_ratio_3_by_uniq', '_ProductID_g2_ratio_3_by_uniq',\
        '_CustomerID_g0_ratio_4_by_uniq', '_CustomerID_g1_ratio_4_by_uniq', '_CustomerID_g2_ratio_4_by_uniq', '_CustomerID_g3_ratio_4_by_uniq',\
        '_CustomerID_g0_ratio_5_by_usetime', '_CustomerID_g1_ratio_5_by_usetime', '_CustomerID_g2_ratio_5_by_usetime',\
        '_CustomerID_g3_ratio_5_by_usetime', '_CustomerID_g4_ratio_5_by_usetime', '_CustomerID_g5_ratio_5_by_usetime', '_CustomerID_g6_ratio_5_by_usetime',\
    '_duration_90_10','_CustomerID_g0_ratio_5_by_time','_CustomerID_g1_ratio_5_by_time','_CustomerID_g2_ratio_5_by_time',\
            '_CustomerID_g3_ratio_5_by_time','_CustomerID_g4_ratio_5_by_time','_PrdocutID_g0_ratio_9', '_PrdocutID_g1_ratio_9', '_PrdocutID_g2_ratio_9',\
             '_PrdocutID_g3_ratio_9','_PrdocutID_g4_ratio_9', '_PrdocutID_g5_ratio_9', '_PrdocutID_g6_ratio_9',\
             '_PrdocutID_g7_ratio_9','_PrdocutID_g8_ratio_9','_CustomerID_g0_ratio_3','_CustomerID_g1_ratio_3',\
             '_CustomerID_g2_ratio_3','CustomerID_ProductID_QueryTS_gt1_count','CustomerID_ProductID_QueryTS_gt1_mean','CustomerID_ProductID_QueryTS_gt1_std','CustomerID_ProductID_QueryTS_gt1_min','CustomerID_ProductID_QueryTS_gt1_25',\
                    'CustomerID_ProductID_QueryTS_gt1_50','CustomerID_ProductID_QueryTS_gt1_70','CustomerID_ProductID_QueryTS_gt1_max','CustomerID_ProductID_QueryTS_gt1_count_gt_up','CustomerID_ProductID_QueryTS_gt1_count_lt_down',\
    '_PrdocutID_g0_ratio', '_PrdocutID_g1_ratio', '_PrdocutID_g2_ratio', '_PrdocutID_g3_ratio','diff_count', 'diff_mean', 'diff_std', 'diff_min', 'diff_25', 'diff_50', 'diff_70', 'diff_max','counts', '_22_01_ratio', '_04_15_ratio', '_weekday_0_ratio', '_weekday_235_ratio','CustomerID_count','CustomerID_mean','CustomerID_std','CustomerID_min','CustomerID_25',\
                    'CustomerID_50','CustomerID_70','CustomerID_max','CustomerID_count_gt_up','CustomerID_count_lt_down','CustomerID_count_gt_one',\
                    'ProductID_count','ProductID_mean','ProductID_std','ProductID_min','ProductID_25',\
                    'ProductID_50','ProductID_70','ProductID_max','ProductID_count_gt_up','ProductID_count_lt_down','ProductID_count_gt_one',\
                    'CustomerID_ProductID_count','CustomerID_ProductID_mean','CustomerID_ProductID_std','CustomerID_ProductID_min','CustomerID_ProductID_25',\
                    'CustomerID_ProductID_50','CustomerID_ProductID_70','CustomerID_ProductID_max','CustomerID_ProductID_count_gt_up','CustomerID_ProductID_count_lt_down','CustomerID_ProductID_count_gt_one',\
                    'CustomerID_ProductID_QueryTS_count','CustomerID_ProductID_QueryTS_mean','CustomerID_ProductID_QueryTS_std','CustomerID_ProductID_QueryTS_min','CustomerID_ProductID_QueryTS_25',\
                    'CustomerID_ProductID_QueryTS_50','CustomerID_ProductID_QueryTS_70','CustomerID_ProductID_QueryTS_max','CustomerID_ProductID_QueryTS_count_gt_up','CustomerID_ProductID_QueryTS_count_lt_down','CustomerID_ProductID_QueryTS_count_gt_one',\

                    'CustomerID_ProductID_QueryDateTime_date_count','CustomerID_ProductID_QueryDateTime_date_mean','CustomerID_ProductID_QueryDateTime_date_std','CustomerID_ProductID_QueryDateTime_date_min','CustomerID_ProductID_QueryDateTime_date_25',\
                    'CustomerID_ProductID_QueryDateTime_date_50','CustomerID_ProductID_QueryDateTime_date_70','CustomerID_ProductID_QueryDateTime_date_max','CustomerID_ProductID_QueryDateTime_date_count_gt_up','CustomerID_ProductID_QueryDateTime_date_count_lt_down','CustomerID_ProductID_QueryDateTime_date_count_gt_one',\

                    'CustomerID_ProductID_QueryDateTime_date_hour_count','CustomerID_ProductID_QueryDateTime_date_hour_mean','CustomerID_ProductID_QueryDateTime_date_hour_std','CustomerID_ProductID_QueryDateTime_date_hour_min','CustomerID_ProductID_QueryDateTime_date_hour_25',\
                    'CustomerID_ProductID_QueryDateTime_date_hour_50','CustomerID_ProductID_QueryDateTime_date_hour_70','CustomerID_ProductID_QueryDateTime_date_hour_max','CustomerID_ProductID_QueryDateTime_date_hour_count_gt_up','CustomerID_ProductID_QueryDateTime_date_hour_count_lt_down','CustomerID_ProductID_QueryDateTime_date_hour_count_gt_one',\

                    'CustomerID_ProductID_diff_count','CustomerID_ProductID_diff_mean','CustomerID_ProductID_diff_std','CustomerID_ProductID_diff_min','CustomerID_ProductID_diff_25',\
                    'CustomerID_ProductID_diff_50','CustomerID_ProductID_diff_70','CustomerID_ProductID_diff_max','CustomerID_ProductID_diff_count_gt_up','CustomerID_ProductID_diff_count_lt_down','CustomerID_ProductID_diff_count_gt_one',\
                    'CustomerID_ProductID_QueryTS_one_file_count','CustomerID_ProductID_QueryTS_one_file_mean','CustomerID_ProductID_QueryTS_one_file_std','CustomerID_ProductID_QueryTS_one_file_min','CustomerID_ProductID_QueryTS_one_file_25',\
                    'CustomerID_ProductID_QueryTS_one_file_50','CustomerID_ProductID_QueryTS_one_file_70','CustomerID_ProductID_QueryTS_one_file_max','CustomerID_ProductID_QueryTS_one_file_count_gt_up','CustomerID_ProductID_QueryTS_one_file_count_lt_down','CustomerID_ProductID_QueryTS_one_file_count_gt_one',\
                    'CustomerID_ProductID_QueryDateTime_date_one_file_count','CustomerID_ProductID_QueryDateTime_date_one_file_mean','CustomerID_ProductID_QueryDateTime_date_one_file_std','CustomerID_ProductID_QueryDateTime_date_one_file_min','CustomerID_ProductID_QueryDateTime_date_one_file_25',\
                    'CustomerID_ProductID_QueryDateTime_date_one_file_50','CustomerID_ProductID_QueryDateTime_date_one_file_70','CustomerID_ProductID_QueryDateTime_date_one_file_max','CustomerID_ProductID_QueryDateTime_date_one_file_count_gt_up','CustomerID_ProductID_QueryDateTime_date_one_file_count_lt_down','CustomerID_ProductID_QueryDateTime_date_one_file_count_gt_one',\
                    'CustomerID_ProductID_QueryDateTime_date_hour_one_file_count','CustomerID_ProductID_QueryDateTime_date_hour_one_file_mean','CustomerID_ProductID_QueryDateTime_date_hour_one_file_std','CustomerID_ProductID_QueryDateTime_date_hour_one_file_min','CustomerID_ProductID_QueryDateTime_date_hour_one_file_25',\
                    'CustomerID_ProductID_QueryDateTime_date_hour_one_file_50','CustomerID_ProductID_QueryDateTime_date_hour_one_file_70','CustomerID_ProductID_QueryDateTime_date_hour_one_file_max','CustomerID_ProductID_QueryDateTime_date_hour_one_file_count_gt_up','CustomerID_ProductID_QueryDateTime_date_hour_one_file_count_lt_down','CustomerID_ProductID_QueryDateTime_date_hour_one_file_count_gt_one',\
                    'CustomerID_ProductID_diff_one_file_count','CustomerID_ProductID_diff_one_file_mean','CustomerID_ProductID_diff_one_file_std','CustomerID_ProductID_diff_one_file_min','CustomerID_ProductID_diff_one_file_25',\
                    'CustomerID_ProductID_diff_one_file_50','CustomerID_ProductID_diff_one_file_70','CustomerID_ProductID_diff_one_file_max','CustomerID_ProductID_diff_one_file_count_gt_up','CustomerID_ProductID_diff_one_file_count_lt_down','CustomerID_ProductID_diff_one_file_count_gt_one',\
                '_duration', '_3_month_ratio', '_4_month_ratio', '_5_month_ratio', '_6_month_ratio',
    # ae
    'ae_0', 'ae_1', 'ae_2', 'ae_3',
    'ae_all_0', 'ae_all_1', 'ae_all_2', 'ae_all_3', 'ae_all_4', 'ae_all_5',
       'ae_all_6', 'ae_all_7', 'ae_all_8', 'ae_all_9',
      ]

query_log_group_by_fileID = pd.read_csv('feature_file/query_log_group_by_fileID_TS_QueryDateTime_date_hour_diff_fuxy_1_weeekday_product_cluster.csv')
query_log_group_by_fileID_duration = pd.read_csv('feature_file/query_log_group_by_fileID_duration.csv')
query_log_group_by_fileID_clustering = pd.read_csv('feature_file/query_log_productID3_customerID9_clustering_with_gird_uniqNusetime_uniq_count_ratio.csv')
query_log_diff = pd.read_csv('feature_file/query_log_diff.csv')
query_log_by_file = pd.read_csv('feature_file/query_log_by_file.csv')
seldom_period = pd.read_csv('feature_file/query_log_seldom_period_diffusion_grad_grad.csv')
query_log_time_to_complete = pd.read_csv('feature_file/query_log_time_to_complete.csv')
query_log_each_product = pd.read_csv('feature_file/query_log_each_product.csv')
query_log_diff_cluster = pd.read_csv('feature_file/query_log_diff_cluster.csv')
cv_customer_ratio = pd.read_csv('feature_file/cv_customer_ratio_5.csv')
cv_product_ratio = pd.read_csv('feature_file/cv_product_ratio.csv')
cv_customer_product_ratio = pd.read_csv('feature_file/cv_customer_product_ratio.csv')
cv_customer_uniq_ratio = pd.read_csv('feature_file/cv_customer_uniq_ratio.csv')
cv_customer_uniq_ratio = pd.read_csv('feature_file/cv_customer_uniq_ratio.csv')
cv_high_virus_customer_diff = pd.read_csv('feature_file/cv_high_virus_customer_diff.csv')
cv_hour_ratio = pd.read_csv('feature_file/cv_hour_ratio.csv')
ae_df = pd.read_csv('feature_file/ae_feature1.csv')
ae_all = pd.read_csv('feature_file/ae_all.csv')
cv_high_virus_customer_diff_with_nonsorted_grad_grad = pd.read_csv('feature_file/cv_high_virus_customer_diff_with_nonsorted_grad_grad.csv')
query_log_seldom_period_diffusion_nonsorted = pd.read_csv('feature_file/query_log_seldom_period_diffusion_nonsorted.csv')
ae_all_0322_epochs1000 = pd.read_csv('feature_file/ae_all_0322_epochs1000.csv')
exception_train = pd.read_csv('feature_file/exception_train.txt', names=['FileID'])
train_full = pd.read_csv('feature_file/training-set.csv', names=['FileID','res'])
train_no_exception = train_full[[i not in [i for i in exception_train.FileID] for i in train_full.FileID]]
train_no_exception = pd.merge(train_no_exception,query_log_group_by_fileID,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_group_by_fileID_duration,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_group_by_fileID_clustering,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_diff,on='FileID')
train_no_exception = pd.merge(train_no_exception,ae_df,on='FileID')
train_no_exception = pd.merge(train_no_exception,seldom_period,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_time_to_complete,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_each_product,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_diff_cluster,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_customer_ratio,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_product_ratio,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_customer_product_ratio,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_customer_uniq_ratio,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_high_virus_customer_diff,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_by_file,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_hour_ratio,on='FileID')
train_no_exception = pd.merge(train_no_exception,ae_all,on='FileID')
train_no_exception = pd.merge(train_no_exception,cv_high_virus_customer_diff_with_nonsorted_grad_grad,on='FileID')
train_no_exception = pd.merge(train_no_exception,query_log_seldom_period_diffusion_nonsorted,on='FileID')
train_no_exception = pd.merge(train_no_exception,ae_all_0322_epochs1000,on='FileID')

train = train_no_exception.fillna(train_no_exception.median())
train_X = train[var]
train_y = train['res']

##### feature selection #####

clf = lgb.LGBMClassifier(seed=2018,max_depth=10)
br_3 = BoostARoota(metric='auc',clf=clf,cutoff=3)
br_3.fit(train_X, train_y)

#Can look at the important variables - will return a pandas series
br_3.keep_vars_

#Then modify dataframe to only include the important variables
br_3.transform(train_X)

br_4 = BoostARoota(metric='auc',clf=clf)
br_4.fit(train_X, train_y)

#Can look at the important variables - will return a pandas series
br_4.keep_vars_

#Then modify dataframe to only include the important variables
br_4.transform(train_X)

br_xgb_3 = BoostARoota(metric='auc',cutoff=3)
br_xgb_3.fit(train_X, train_y)
#Can look at the important variables - will return a pandas series
br_xgb_3.keep_vars_
#Then modify dataframe to only include the important variables
br_xgb_3.transform(train_X)


br_xgb_4 = BoostARoota(metric='auc')
br_xgb_4.fit(train_X, train_y)

#Can look at the important variables - will return a pandas series
br_xgb_4.keep_vars_

#Then modify dataframe to only include the important variables
br_xgb_4.transform(train_X)


##### model tuning #####
model_lgb_3 = lgb.LGBMClassifier(seed=2018,num_leaves=60,learning_rate=0.05,max_bin=150,
                        min_data_in_leaf=20,lambda_l1=0.25,bagging_freq=10)
parameters = {'max_depth':range(7,12), 'n_estimators':[700,800,900,1000,1100]}
model_lgb_3 = GridSearchCV(model_lgb_3, parameters,cv=3)
model_lgb_3.fit(br_3.transform(train_X), train_y.values)


# xgb
a = br_xgb_3.transform(train_X)
params = {
        'min_child_weight': [1, 5, 10],
        'max_depth': [5, 7, 10]
        }
folds = 3
model_xgb_3 = XGBClassifier(learning_rate=0.05, n_estimators=600,
                    silent=True, nthread=4,subsample=0.8,gamma=1.5)
model_xgb_3 = GridSearchCV(model_xgb_3, parameters,cv=3)
model_xgb_3.fit(a.values, train_y.values)

a = br_xgb_4.transform(train_X)
params = {
        'min_child_weight': [1, 5, 10],
        'max_depth': [5, 7, 10]
        }
folds = 3
model_xgb_4 = XGBClassifier(learning_rate=0.05, n_estimators=600,
                    silent=True, nthread=4,subsample=0.8,gamma=1.5)
model_xgb_4 = GridSearchCV(model_xgb_4, params,cv=3)
model_xgb_4.fit(a.values, train_y.values)

##### model tuning #####


##### feature selection #####



train_lgb_3 = br_3.transform(train_X)
lgb3 = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in train_lgb_3.columns])),
                                    lgb.LGBMClassifier(bagging_freq=10, boosting_type='gbdt', colsample_bytree=1.0,
                                            lambda_l1=0.25, learning_rate=0.05, max_bin=150, max_depth=9,
                                            min_child_samples=10, min_child_weight=5, min_data_in_leaf=20,
                                            min_split_gain=0.0, n_estimators=800, n_jobs=-1, num_leaves=60,
                                            objective=None, random_state=0, reg_alpha=0.0, reg_lambda=0.0,
                                            seed=2018, silent=True, subsample=1.0, subsample_for_bin=50000,
                                            subsample_freq=1))

train_xgb_3 = br_xgb_3.transform(train_X)
xgb3 = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in train_xgb_3.columns])),
                                    xgb.XGBClassifier(learning_rate=0.05, n_estimators=600,
                    silent=True, nthread=4,subsample=0.8,gamma=1.5, **model_xgb_3.best_params_))

train_lgb_3 = br_3.transform(train_X)
lgb3 = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in train_lgb_3.columns])),
                                    lgb.LGBMClassifier(bagging_freq=10, boosting_type='gbdt', colsample_bytree=1.0,
                                            lambda_l1=0.25, learning_rate=0.05, max_bin=150, max_depth=9,
                                            min_child_samples=10, min_child_weight=5, min_data_in_leaf=20,
                                            min_split_gain=0.0, n_estimators=800, n_jobs=-1, num_leaves=60,
                                            objective=None, random_state=0, reg_alpha=0.0, reg_lambda=0.0,
                                            seed=2018, silent=True, subsample=1.0, subsample_for_bin=50000,
                                            subsample_freq=1))

train_xgb_4 = br_xgb_4.transform(train_X)
xgb4 = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in train_xgb_4.columns])),
                    xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                           gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=7,
                           min_child_weight=1, missing=None, n_estimators=1100, nthread=4,
                           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                           scale_pos_weight=1, seed=0, silent=True, subsample=1)
                    )

train_lgb_4 = br_4.transform(train_X)
lgb4 = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in train_lgb_4.columns])),
                                    lgb.LGBMClassifier(bagging_freq=10, boosting_type='gbdt', colsample_bytree=1.0,
                                            lambda_l1=0.25, learning_rate=0.05, max_bin=150, max_depth=10,
                                            min_child_samples=10, min_child_weight=5, min_data_in_leaf=20,
                                            min_split_gain=0.0, n_estimators=600, n_jobs=-1, num_leaves=60,
                                            objective=None, random_state=0, reg_alpha=0.0, reg_lambda=0.0,
                                            seed=2018, silent=True, subsample=1.0, subsample_for_bin=50000,
                                            subsample_freq=1))


xgb_best = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in var_best])),
                                    xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=7,
       min_child_weight=1, missing=None, n_estimators=1100, nthread=4,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1))


lgbbest = make_pipeline(ColumnSelector(cols=
                                     tuple([train_X.columns.get_loc(c) for c in train_X.columns if c in var_best])),
                                    lgb.LGBMClassifier(bagging_freq=10, boosting_type='gbdt', colsample_bytree=1.0,
        learning_rate=0.05, max_bin=255, max_depth=-1,
        min_child_samples=10, min_child_weight=5, min_split_gain=0.0,
        n_estimators=10, n_jobs=-1, num_leaves=60, objective=None,
        random_state=0, reg_alpha=0.0, reg_lambda=0.0, seed=2018,
        silent=True, subsample=1.0, subsample_for_bin=50000,
        subsample_freq=1))

eclf3 = VotingClassifier(estimators=[
    ('lgb3', lgb3), ('xgb3', xgb3), ('xgb4', xgb4), ('xgb_best', xgb_best), ('lgbbest', lgbbest)],
    voting='soft',
    flatten_transform=True)

eclf3.fit(train_X.values, train_y.values)


test_full = pd.read_csv('feature_file/testing-set.csv', names=['FileID','Probability'])
exception_test = pd.read_csv('feature_file/exception_testing.txt', names=['FileID'])
query_log_group_by_fileID_clustering = pd.read_csv('feature_file/query_log_productID3_customerID9_clustering_with_gird_uniqNusetime_uniq_count_ratio.csv')
query_log_diff = pd.read_csv('feature_file/query_log_diff.csv')
ae_df = pd.read_csv('feature_file/ae_feature1.csv')
seldom_period = pd.read_csv('feature_file/query_log_seldom_period_diffusion_grad_grad.csv')
query_log_time_to_complete = pd.read_csv('feature_file/query_log_time_to_complete.csv')
query_log_diff_cluster = pd.read_csv('feature_file/query_log_diff_cluster.csv')
query_log_each_product = pd.read_csv('feature_file/query_log_each_product.csv')
cv_customer_ratio_test = pd.read_csv('feature_file/cv_customer_ratio_test.csv')

cv_product_ratio_test = pd.read_csv('feature_file/cv_product_ratio_test.csv')
cv_customer_product_ratio_test = pd.read_csv('feature_file/cv_customer_product_ratio_test.csv')
cv_customer_uniq_ratio_test = pd.read_csv('feature_file/cv_customer_uniq_ratio_test.csv')
cv_high_virus_customer_diff_test = pd.read_csv('feature_file/cv_high_virus_customer_diff_test.csv')
cv_hour_ratio_test = pd.read_csv('feature_file/cv_hour_ratio_test.csv')
query_log_by_file = pd.read_csv('feature_file/query_log_by_file.csv')

query_log_by_file = pd.read_csv('feature_file/query_log_by_file.csv')
ae_all_test = pd.read_csv('feature_file/ae_all_test.csv')
ae_all_0322_epochs1000_test = pd.read_csv('feature_file/ae_all_0322_epochs1000_test.csv')

cv_high_virus_customer_diff_test_with_nonsorted_grad_grad = pd.read_csv('feature_file/cv_high_virus_customer_diff_test_with_nonsorted_grad_grad.csv')
query_log_seldom_period_diffusion_nonsorted = pd.read_csv('feature_file/query_log_seldom_period_diffusion_nonsorted.csv')


test = test_full[[i not in [i for i in exception_test.FileID] for i in test_full.FileID]]
test = pd.merge(test,ae_df,on='FileID')

test = pd.merge(test,query_log_group_by_fileID,on='FileID')
test = pd.merge(test,query_log_group_by_fileID_duration,on='FileID')
test = pd.merge(test,query_log_group_by_fileID_clustering,on='FileID')
test = pd.merge(test,query_log_diff,on='FileID')
test = pd.merge(test,seldom_period,on='FileID')
test = pd.merge(test,query_log_time_to_complete,on='FileID')
test = pd.merge(test,query_log_each_product,on='FileID')
test = pd.merge(test,query_log_diff_cluster,on='FileID')
test = pd.merge(test,cv_customer_ratio_test,on='FileID')

test = pd.merge(test,cv_product_ratio_test,on='FileID')
test = pd.merge(test,cv_customer_product_ratio_test,on='FileID')
test = pd.merge(test,cv_customer_uniq_ratio_test,on='FileID')
test = pd.merge(test,cv_high_virus_customer_diff_test,on='FileID')
test = pd.merge(test,cv_hour_ratio_test,on='FileID')
test = pd.merge(test,query_log_by_file,on='FileID')

test = pd.merge(test,ae_all_test,on='FileID')

test = pd.merge(test,cv_high_virus_customer_diff_test_with_nonsorted_grad_grad,on='FileID')
test = pd.merge(test,query_log_seldom_period_diffusion_nonsorted,on='FileID')
test = pd.merge(test,ae_all_0322_epochs1000_test,on='FileID')

test = test.fillna(train_no_exception.median())
test_X = test[var]
test_y = test['Probability']
submit = test[['FileID','Probability']]

Probability = eclf3.predict_proba(test_X.values)
submit['Probability'] = pd.DataFrame(Probability)[1]
submit.to_csv('submit_final.csv',index=False)
