import numpy as np
import pandas as pd

import copy
from tqdm import tqdm

def min_max_scale(x):
    cols = x.shape[1]
    min_record = []
    max_record = []

    for col in range(cols):
        min_val= np.min(x[:,col])
        max_val = np.max(x[:,col])
        x[:,col] = (x[:,col] - min_val)/(max_val - min_val)
        min_record.append(min_val)
        max_record.append(max_val)

    return x, min_record,max_record

def zero_score_scale(x):
    cols = x.shape[1]
    for col in range(cols):
        x[:,col] = (x[:,col]-np.mean(x[:,col]))/(np.std(x[:,col]))

    return x

def min_max_recover(X, min_vec, max_vec):
    cols = X.shape[1]
    for col in range(cols):
        X[:,col] = X[:,col]*(max_vec[col]-min_vec[col])+min_vec[col]
    return X


NORMALIZERS = {'min_max':min_max_scale,
                'zero_score':zero_score_scale}

RECOVER = {'min_max':min_max_recover}

class Solver(object):
    def __init__(self,
                 init_fill_method='zero',
                 normalizer = None,
                 ):

        self.fill_method = init_fill_method
        self.normalizer = normalizer

    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:  # Note that ndarray's shpe is a tuple like (rows, cols)
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing_mask):
        """
        check whether your wait-imputation data contains null value
        :param missing: missing totally as your 'mask', an numpy array.
        :return:raise error
        """
        if isinstance(missing_mask, pd.DataFrame):
            missing_mask = missing_mask.values
        if not missing_mask.any():
            raise ValueError("Input matrix is not missing any values")
        if missing_mask.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _judge_type(self, X):
        coltype_dic = {}
        for col in range(X.shape[1]):
            col_val = X[:, col]
            nan_index = np.where(pd.isnull(col_val))
            col_val = np.delete(col_val, nan_index)

            if len(np.unique(col_val)) / len(col_val) < 0.05 and (np.any(col_val == col_val.astype(int))):
                coltype_dic[col] = 'categorical'
            else:
                coltype_dic[col] = 'continuous'
        return coltype_dic

    def solve(self, X,missing_mask):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def complete(self, x):
        """
        Expects 2d float matrix with NaN entries signifying missing values
        Returns completed matrix without any NaNs.
        """
        self._check_input(x)
        self._check_missing_value_mask(pd.isnull(x))
        x, missing_mask = self.prepare_input_data(x)

        x_zero_replaced = self.fill(x.copy(),missing_mask,'zero')
        if self.normalizer is not None:
            normalizer = NORMALIZERS[self.normalizer]
            x_zero_replaced, min_record, max_record = normalizer(x_zero_replaced)

        x_filled = self.solve(x_zero_replaced, missing_mask)
        revocer = RECOVER[self.normalizer]
        x_filled = revocer(x_filled, min_record, max_record)
        return x_filled


    def sort_col(self, mask):
        """
        count various cols, the missing value wages,
        :param X: the original data matrix which is waiting to be imputed
        :return: col1, col2,.... colx, those cols has been sorted according its status of missing values
        """
        nan_index = np.where(mask == True)[1]
        unique = np.unique(nan_index)
        nan_index = list(nan_index)
        dict = {}
        for item in unique:
            count = nan_index.count(item)
            dict[item] = count
        tmp = sorted(dict.items(), key=lambda e: e[1], reverse=True)
        sort_index = []
        for item in tmp:
            sort_index.append(item[0])
        return sort_index

    def get_type_index(self, mask_all, col_type_dict):
        """
        get the index of every missing value, because the imputed array is 1D
        where the continuous and categorical index are needed.
        :param mask_all:
        :param col_type_dict:
        :return: double list
        """
        where_target = np.argwhere(mask_all == True)
        imp_categorical_index = []
        imp_continuous_index = []
        for index in where_target:
            col_type = col_type_dict[index[1]]
            if col_type == 'categotical':
                imp_categorical_index.append(index)
            elif col_type == 'continuous':
                imp_continuous_index.append(index)

        return imp_continuous_index, imp_categorical_index

    @staticmethod
    def _fill_column_with_fn(X, missing_mask, method):
        """
        :param X: numpy array, the data which waiting to be imputation
        :param missing_mask:numpy array
        :param method: the way of what kind of normal imputation algorithm you use
        :return:
        """
        n_missing = missing_mask.sum()  # np.sum() which could calculate the number of 'TRUE'
        if n_missing == 0:
            return X

        if method == 'frequency':
            unique, counts = np.array(np.unique(X[~np.isnan(X)], return_counts=True))
            fill_values = np.random.choice(unique, size=np.count_nonzero(np.isnan(X)), p=counts / np.sum(counts))
        else:
            fill_values = method(X)
        X[missing_mask] = fill_values

        return X

    def fill(self, X, missing_mask, fill_method=None):
        """
        Parameters
        ----------
        X : np.array or pandas.DataFrame
            Data array containing NaN entries
        missing_mask : np.array
            Boolean array indicating where NaN entries are
            matrix like: [[T,F,T T],
                          [F,T,T,T]
                          [.......]]
        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column
        inplace : bool
            Modify matrix or fill a copy
        """
        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random", "frequency"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0  # this is the match data feature of numpy array
        elif fill_method == "mean":
            self._fill_column_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_column_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_column_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "frequency":
            self._fill_column_with_fn(X, missing_mask, "frequency")
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = np.asarray(X)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def split(self, X, target_col, mask):
        col_mask = mask[:,target_col]
        nan_index = np.where(col_mask == True)
        not_nan_index = np.where(col_mask == False)

        contain_nan_rows = np.delete(X, not_nan_index, 0)
        no_contain_nan_rows = np.delete(X, nan_index, 0)

        train_X = np.delete(no_contain_nan_rows, target_col, 1)
        train_y = no_contain_nan_rows[:, target_col]
        test_X = np.delete(contain_nan_rows, target_col, 1)

        return train_X, train_y, test_X

    @staticmethod
    def _get_missing_loc(missing_mask):
        missing_tuple = np.where(missing_mask)
        missing_row = missing_tuple[0]
        missing_col = missing_tuple[1]
        location = zip(missing_row, missing_col)
        return location, missing_row, missing_col

    @staticmethod
    def _pure_data(data, missing_mask):
        """
        pure a completely data set from data
        :param data: a matrix which contains missing value
        :param missing_mask:
        :return: a complete data set
        """
        missing_rows = np.where(missing_mask)[0]
        pure_data = np.delete(data, missing_rows, axis=0)

        return pure_data

    def _is_mix_type(self, X):
        mask_dict = self.masker(X)
        categorical_count = 0
        continuous_count = 0
        for col in range(X.shape[1]):
            col_type = mask_dict[col]
            if col_type == 'categotical':
                categorical_count += 1
            elif col_type == 'continuous':
                continuous_count += 1

        if categorical_count == 0 and continuous_count != 0:
            return 'continuous'
        elif categorical_count != 0 and continuous_count == 0:
            return 'categotical'
        elif categorical_count != 0 and continuous_count != 0:
            return 'mix'
        else:
            raise ("unkonwn col type")

    @staticmethod
    def extract_imp_data_by_col(filled_X, required_cols, col_mask):
        x_imp = []
        for col in required_cols:
            nan_val = filled_X[:, col][col_mask[col]]
            for item in nan_val:
                x_imp.append(item)
        return x_imp

    def detect_complete_part(self, missing_mask):
        complete_rows = []
        missing_rows = []
        for idx, row in enumerate(missing_mask):
            if (row == False).all():
                complete_rows.append(idx)
            else:
                missing_rows.append(idx)

        return np.asarray(complete_rows), np.asarray(missing_rows)

def generate_noise(n_rows, n_cols):
    """
    generate noise matrix
    """
    return np.random.uniform(0., 1., size=[n_rows, n_cols])

class EM(Solver):
    """
    this algorithm just require to lean the Gauss distribution elements 'mu' and 'sigma'
    """
    def __init__(self,
                 max_iter=1,
                 theta=1e-9,
                 normalizer='min_max'):
        Solver.__init__(self,
                        normalizer=normalizer)

        self.max_iter = max_iter
        self.theta = theta

    def _init_parameters(self, X):
        rows, cols = X.shape
        mu_init = np.nanmean(X, axis=0)
        sigma_init = np.zeros((cols, cols))
        for i in range(cols):
            for j in range(i, cols):
                vec_col = X[:, [i, j]]
                vec_col = vec_col[~np.any(np.isnan(vec_col), axis=1), :].T
                if len(vec_col) > 0:
                    cov = np.cov(vec_col)
                    cov = cov[0, 1]
                    sigma_init[i, j] = cov
                    sigma_init[j, i] = cov

                else:
                    sigma_init[i, j] = 1.0
                    sigma_init[j, i] = 1.0

        return mu_init, sigma_init

    def _e_step(self, mu,sigma, X):
        samples,_ = X.shape
        for sample in range(samples):
            if np.any(np.isnan(X[sample,:])):
                loc_nan = np.isnan(X[sample,:])
                new_mu = np.dot(sigma[loc_nan, :][:, ~loc_nan],
                                np.dot(np.linalg.inv(sigma[~loc_nan, :][:, ~loc_nan]),
                                       (X[sample, ~loc_nan] - mu[~loc_nan])[:,np.newaxis]))
                nan_count = np.sum(loc_nan)
                X[sample, loc_nan] = mu[loc_nan] + new_mu.reshape(1,nan_count)

        return X

    def _m_step(self,X):
        rows, cols = X.shape
        mu = np.mean(X, axis=0)
        sigma = np.cov(X.T)
        tmp_theta = -0.5 * rows * (cols * np.log(2 * np.pi) +
                                  np.log(np.linalg.det(sigma)))

        return mu, sigma,tmp_theta



    def solve(self, X, missing_mask):
        mu, sigma = self._init_parameters(X)
        complete_X,updated_X = None, None
        rows,_ = X.shape
        theta = -np.inf
        for iter in tqdm(range(self.max_iter)):
            updated_X = self._e_step(mu=mu, sigma=sigma, X=copy.copy(X))
            mu, sigma, tmp_theta = self._m_step(updated_X)
            for i in range(rows):
                tmp_theta -= 0.5 * np.dot((updated_X[i, :] - mu),
                                          np.dot(np.linalg.inv(sigma), (updated_X[i, :] - mu)[:, np.newaxis]))
            if abs(tmp_theta-theta)<self.theta:
                complete_X = updated_X
                break
            else:
                theta = tmp_theta
        else:
            complete_X = updated_X

        return complete_X