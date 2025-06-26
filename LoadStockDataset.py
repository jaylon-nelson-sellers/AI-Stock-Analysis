import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS
from sklearn.decomposition import FastICA, PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

from IGTD import min_max_transform, select_features_by_variation, table_to_image


class LoadStockDataset:
    """
    A class to load, preprocess, normalize, and select features from a dataset.
    """

    def __init__(self, dataset_index, normalize=1, verbose=0):
        """
        Initializes the dataset by loading files, normalizing features, and selecting features based on the parameters.
        """
        # Print loading message if verbose
        if verbose:
            print("Loading File")
        # need to change
        if normalize:
            self.observed = dataset_index
            self.feats = pd.read_csv("feats.csv")
            #NOTE Deletes DATE column
            self.normalize()
            self.feats.to_csv("feats_n.csv", index=False)
            self.targets = pd.read_csv("regress.csv")
            return

        self.observed = dataset_index
        self.feats = pd.read_csv("feats_n.csv")
        self.targets = pd.read_csv("regress.csv")


        # Read classification, features, and regression targets
        # change back after vae

        # Select targets based on the target index
        if False:
            # Replace missing and infinite values with zero
            self.feats.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
            self.targets.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

            def convert_to_numeric(df):
                return df.apply(pd.to_numeric, errors='coerce').fillna(0)

            self.feats = convert_to_numeric(self.feats)
            self.targets = convert_to_numeric(self.targets)

            # Normalize features if requested

            # Replace missing and infinite values with zero
            self.feats.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
            self.targets.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
            #
            #print("Dataset Loaded")

    def get_train_test_split(self, split=0.2):
        """
        Splits the features and targets into training and testing sets.
        """
        return train_test_split(self.feats, self.targets, test_size=split, random_state=1)

    def is_power_of_two(self, n):
        return (n != 0) and (n & (n - 1) == 0)

    def next_smallest_power_of_two(self, n):
        if n <= 1:
            return 0
        return 2 ** ((n - 1).bit_length() - 1)

    def confirm_power_of_two(self, feats_3d):
        num_columns = feats_3d.shape[2]

        if not self.is_power_of_two(num_columns):
            num_columns = feats_3d.shape[2]
            next_pow2 = self.next_smallest_power_of_two(num_columns)
            additional_cols = next_pow2 - num_columns
            random_array = np.random.rand(feats_3d.shape[0], feats_3d.shape[1], additional_cols)
            feats_3d = np.concatenate((feats_3d, random_array), axis=2)
        return feats_3d

    def getTesting(self):
        feats = self.feats.tail(1).copy()
        observed = self.observed + 1
        # Add empty columns if necessary
        num_columns = feats.shape[1]
        remainder = num_columns % observed
        if remainder != 0:
            num_extra_columns = observed - remainder
            print(f"Adding {num_extra_columns} to fill Dataframe")
            # Perform FastICA with num_extra_columns components
            ica = FastICA(n_components=num_extra_columns, random_state=0)
            ica_components = ica.fit_transform(feats)

            # Add the FastICA components as new columns to feats
            for i in range(num_extra_columns):
                feats[f'ica_{i}'] = ica_components[:, i]
        self.feats = feats
        # self.feats.to_csv(f"feats_days-{self.observed}_comp-{self.feats.shape[1]}-v2.csv", index=False)

        # Reshape the dataframe to 3D
        new_depth = observed
        new_shape = (feats.shape[0], feats.shape[1] // new_depth, new_depth)
        feats_3d = feats.values.reshape(new_shape)
        feats_3d = np.swapaxes(feats_3d, 1, 2)
        return feats_3d

    def getIGTD(self):
        with open("IGTD\Results.pkl", 'rb') as picklefile:
            feats = pickle.load(picklefile)
            data_condition2 = np.transpose(feats, (2, 0, 1))[:, np.newaxis, :, :]
            print("Shape of feats:", data_condition2.shape)
            self.feats = data_condition2
            print(self.feats.shape)

    def IGTD(self):          
        """
        IGTD Function for data augmentation.
        @param self reference
        @sum
        """
        print(math.sqrt(self.feats.shape[1]) * self.observed)
        num_columns = int(math.sqrt(self.feats.shape[1]) * self.observed)
        if not self.is_power_of_two(num_columns):
            next_pow2 = self.next_smallest_power_of_two(num_columns)
        print(next_pow2)
        num_row = next_pow2  # Number of pixel rows in image representation
        num_col = next_pow2  # Number of pixel columns in image representation
        num = num_row * num_col  # Number of features to be included for analysis, which is also the total number of pixels in image representation
        save_image_size = 6  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
        max_step = 30000  # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
        val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
        # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

        # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
        data = self.feats
        # Select features with large variations across samples
        id = select_features_by_variation(data, variation_measure='var', num=num)
        data = data.iloc[:, id]
        # Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
        norm_data = min_max_transform(data.values)
        norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

        # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
        # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
        # the pixel distance ranking matrix. Save the result in Test_1 folder.
        fea_dist_method = 'Euclidean'
        image_dist_method = 'Euclidean'
        error = 'abs'
        result_dir = 'IGTD'
        os.makedirs(name=result_dir, exist_ok=True)
        table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                       max_step, val_step, result_dir, error)
        print('IGTD Complete')

    def get_3d(self, version=1, split=.2):
        if version == 0:
            # "1D" Convolution
            # [n_samples, 1, n_features]
            new_shape = (self.feats.shape[0], 1, self.feats.shape[1])
            self.feats = self.feats.values.reshape(new_shape)
            return train_test_split(self.feats, self.targets, test_size=split, random_state=1)

        if version == 1:
            # 2D Method 1: Stacking Observed Days
            # [n_samples,observed_days,n_features/observed_days]
            if self.observed == 1:

                new_shape = (self.feats.shape[0], 1, self.feats.shape[1])
                self.feats = self.feats.values.reshape(new_shape)
                return train_test_split(self.feats, self.targets, test_size=split, random_state=1)
            else:
                feats = self.feats
                observed = self.observed + 1
                # Add empty columns if necessary
                num_columns = feats.shape[1]
                remainder = num_columns % observed
                if remainder != 0:
                    num_extra_columns = observed - remainder
                    print(f"Adding {num_extra_columns} to fill Dataframe")
                    # Perform FastICA with num_extra_columns components
                    ica = FastICA(n_components=num_extra_columns, random_state=0)
                    ica_components = ica.fit_transform(feats)

                    # Add the FastICA components as new columns to feats
                    for i in range(num_extra_columns):
                        feats[f'ica_{i}'] = ica_components[:, i]
                self.feats = feats
                # self.feats.to_csv(f"feats_days-{self.observed}_comp-{self.feats.shape[1]}-v2.csv", index=False)

                # Reshape the dataframe to 3D
                new_depth = observed
                new_shape = (feats.shape[0], feats.shape[1] // new_depth, new_depth)
                feats_3d = feats.values.reshape(new_shape)
                feats_3d = np.swapaxes(feats_3d, 1, 2)
                self.feats = feats_3d
                print(f"New Shape:{self.feats.shape}")
                return train_test_split(self.feats, self.targets, test_size=split, random_state=1)
        if version == 2:
            # need to finish this part
            self.IGTD()
            self.getIGTD()
            return train_test_split(self.feats, self.targets, test_size=split, random_state=1)

    from sklearn.preprocessing import StandardScaler

    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    import numpy as np

    def normalize(self):
        # Create a copy of the original dataframe
        normalized_df = self.feats

        # Initialize RobustScaler
        scaler = StandardScaler()

        normalized_data = scaler.fit_transform(normalized_df)

        self.feats = pd.DataFrame(normalized_data)


    def apply_clustering(self):
        """
        Apply multiple clustering methods with varying numbers of clusters
        and append the results to the original dataframe.

        :param df: Original dataframe
        :param feature_columns: List of column names to use for clustering
        :return: Dataframe with cluster labels appended
        """
        # Extract features for clustering
        features = self.feats


        cluster_labels = {}

        # Apply K-Means and HAC for 2-5 clusters
        for n_clusters in range(2,11):
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(features)
            cluster_labels[f'kmeans_{n_clusters}'] = kmeans_labels



        # Append cluster labels to the original dataframe
        for method, labels in cluster_labels.items():
            features[f'cluster_{method}'] = labels
        self.feats = features

#ld = LoadStockDataset(1,1)