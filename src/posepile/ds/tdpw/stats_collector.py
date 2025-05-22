import numpy as np


class StatCollector():
    def __init__(self, ssq_keys=None, prod_key_pairs=None):
        self.sums = SumDict()
        self.sums_of_squares = SumDict()
        self.sums_of_products = SumDict()
        self.counts = SumDict()
        self.ssq_keys = ssq_keys
        self.prod_key_pairs = prod_key_pairs

    def update(self, dict):
        for k, v in dict.items():
            self.sums[k] += np.sum(v, axis=0)
            self.counts[k] += len(v)

        if self.ssq_keys is not None:
            for k in self.ssq_keys:
                if k in dict:
                    self.sums_of_squares[k] += np.sum(dict[k] ** 2, axis=0)

        if self.prod_key_pairs is not None:
            for k1, k2 in self.prod_key_pairs:
                if k1 in dict and k2 in dict:
                    sumprod = np.sum(dict[k1] * dict[k2], axis=0)
                    self.sums_of_products[sorted_pair(k1, k2)] += sumprod

    def aggregate(self, s, k, aggregate_axes=None):
        if aggregate_axes is None:
            return s[k]
        return np.sum(s[k], axis=aggregate_axes)

    def get_count(self, k, aggregate_axes=None):
        if aggregate_axes is None:
            return self.counts[k]
        n_aggregated = np.prod(np.array(self.sums[k].shape)[aggregate_axes])
        return self.counts[k] * n_aggregated

    def get_sum(self, k, aggregate_axes=None):
        return self.aggregate(self.sums, k, aggregate_axes)

    def get_sum_of_squares(self, k, aggregate_axes=None):
        return self.aggregate(self.sums_of_squares, k, aggregate_axes)

    def get_sum_of_products(self, k1, k2, aggregate_axes=None):
        return self.aggregate(self.sums_of_products, sorted_pair(k1, k2), aggregate_axes)

    def get_mean(self, k, aggregate_axes=None):
        return self.get_sum(k, aggregate_axes) / self.get_count(k, aggregate_axes)

    def get_var(self, k, aggregate_axes=None):
        n = self.get_count(k, aggregate_axes)
        mean_Xsq = self.get_sum_of_squares(k, aggregate_axes) / n
        mean_X = self.get_sum(k, aggregate_axes) / n
        return mean_Xsq - mean_X ** 2

    def get_std(self, k, aggregate_axes=None):
        return np.sqrt(self.get_var(k, aggregate_axes))

    def get_sem(self, k, aggregate_axes=None):
        return self.get_std(k, aggregate_axes) / np.sqrt(self.get_count(k, aggregate_axes))

    def get_corr(self, k1, k2, aggregate_axes=None):
        n = self.get_count(k1, aggregate_axes)
        mean_X = self.get_sum(k1, aggregate_axes) / n
        mean_Y = self.get_sum(k2, aggregate_axes) / n
        mean_XY = self.get_sum_of_products(k1, k2, aggregate_axes) / n
        mean_Xsq = self.get_sum_of_squares(k1, aggregate_axes) / n
        mean_Ysq = self.get_sum_of_squares(k2, aggregate_axes) / n
        numerator = mean_XY - mean_X * mean_Y
        denominator = np.sqrt((mean_Xsq - mean_X ** 2) * (mean_Ysq - mean_Y ** 2))
        return numerator / denominator


class SumDict():
    class ReturnOnIadd():
        def __iadd__(self, other):
            return np.asarray(other, np.float64)

    def __init__(self):
        self.sumd = {}
        self.storer = SumDict.ReturnOnIadd()

    def __getitem__(self, k):
        if k in self.sumd:
            return self.sumd[k]
        else:
            return self.storer

    def __setitem__(self, k, v):
        self.sumd[k] = v


def sorted_pair(a, b):
    return tuple(sorted([a, b]))
