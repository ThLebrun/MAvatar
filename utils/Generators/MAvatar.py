from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

# from utils.utils import dict_loader, patch_nan, func_loader


def compute_kde(data_row, std_dev=500):
    kde = gaussian_kde(data_row)

    def kde_func(
        x,
    ):
        return kde(x)

    return kde_func


def build_generator(arr):
    x = np.linspace(np.min(arr), np.max(arr), 1000)

    # Compute KDEs in parallel
    kde_estimator = compute_kde(arr)
    inverse_cdf = compute_inverse_cdf(kde_estimator, x)
    return inverse_cdf


def compute_inverse_cdf(kde_sum, x):
    cdf = np.cumsum(kde_sum(x))
    cdf = cdf / cdf[-1]  # Normalize to 0-1 range
    inverse_cdf = interp1d(
        cdf, x, bounds_error=False, fill_value=(x[0], x[-1])
    )
    return inverse_cdf


def get_closest_indices(distribution, k_values):
    dist = np.abs(distribution[:, np.newaxis] - k_values[np.newaxis, :])
    label = np.argmin(dist, axis=1)
    labels = np.unique(label)
    return [list(np.argwhere(label == id_label)[:, 0]) for id_label in labels]


def split_by_buckets(intervals, generator_func, distribution):
    """return a list of indexs of elements from distribution where they are the closest form the centers of intervals"""
    mid_values = [np.mean(interval) for interval in intervals]
    k_values = np.array(
        [generator_func(val) for val in mid_values]
    )  # Example k values
    return get_closest_indices(distribution, k_values)


def MAvatar(
    loader,
    seed=None,
    list_ids=[],
    hparams={"k": 20, "nf": -1, "size_ratio": 1},
):
    import saiph

    range_ = np.arange(0, 95, 5)
    intervals = [
        (b_inf / 100, b_sup / 100) for b_inf, b_sup in zip(range_, range_ + 10)
    ]
    data = loader(object_bool=False, list_ids=list_ids).copy()
    np.random.seed(seed)
    if len(list_ids) < 1:
        list_ids = list(range(len(data)))
    data.reset_index(drop=True, inplace=True)
    len_new_data = round(hparams["size_ratio"] * len(list_ids))
    # data = patch_nan(data)
    assert not data.isnull().values.any(), "Dataset contains NaN values!"
    coord, model = saiph.fit_transform(data)
    generator_axis1 = build_generator(coord.iloc[:, 0].to_numpy())

    generator_axis2 = []
    split_coords = [
        [None for i_interval, interval in enumerate(intervals)]
        for i_interval, interval in enumerate(intervals)
    ]
    for i_interval, interval in enumerate(intervals):
        x_min, x_max = generator_axis1(interval[0]), generator_axis1(
            interval[1]
        )
        array = coord.iloc[:, 0].to_numpy()
        where_bucket = np.argwhere((array >= x_min) * (array <= x_max))[:, 0]
        data_bucket = coord.iloc[where_bucket, 1].to_numpy()

        generator_axis2.append(build_generator(data_bucket))
        for j_interval, interval in enumerate(intervals):
            y_min, y_max = generator_axis2[-1](interval[0]), generator_axis2[
                -1
            ](interval[1])
            condition_x = (coord.iloc[:, 0] >= x_min) & (
                coord.iloc[:, 0] <= x_max
            )
            condition_y = (coord.iloc[:, 1] >= y_min) & (
                coord.iloc[:, 1] <= y_max
            )
            split_coords[i_interval][j_interval] = np.where(
                condition_x & condition_y
            )[0]

    arr_coord = coord.to_numpy()

    # indices = np.random.randint(n, size=(m, k))
    indices = np.random.randint(
        coord.shape[0], size=(len_new_data, coord.shape[1])
    )
    new_data = arr_coord[indices, np.arange(coord.shape[1])]

    distribution = generator_axis1(
        np.random.rand(len_new_data)
    )  # Large distribution of floats
    new_data[:, 0] = distribution

    sublists = split_by_buckets(intervals, generator_axis1, distribution)

    # # conditionaly generating second axis
    # for j_interval, interval in enumerate(intervals):
    #     generated_2nd_axis = generator_axis2[j_interval](np.random.rand(len(sublists[j_interval])))
    #     new_data[sublists[j_interval], 1] = np.array(generated_2nd_axis)

    for i_interval, _ in enumerate(intervals):
        sub_points = np.array(sublists[i_interval])
        generated_2nd_axis = generator_axis2[i_interval](
            np.random.rand(len(sublists[i_interval]))
        )
        new_data[sub_points, 1] = np.array(generated_2nd_axis)
        sub_sub_list = split_by_buckets(
            intervals, generator_axis2[i_interval], generated_2nd_axis
        )
        # print(len(sub_sub_list))
        for j_interval, _ in enumerate(sub_sub_list):
            sub_sub_points = np.array(sub_points[sub_sub_list[j_interval]])
            sub_coord_real_tail = coord.iloc[
                split_coords[i_interval][j_interval], 2:
            ].to_numpy()
            if sub_coord_real_tail.shape[0] > 0:
                indices = np.random.randint(
                    sub_coord_real_tail.shape[0],
                    size=(len(sub_sub_points), sub_coord_real_tail.shape[1]),
                )
                new_data[sub_sub_points, 2:] = sub_coord_real_tail[
                    indices, np.arange(sub_coord_real_tail.shape[1])
                ]
            else:
                pass

    # print(new_data.columns)
    new_data = pd.DataFrame(new_data)
    new_data = saiph.inverse_transform(new_data, model)

    return new_data  # func_loader[pb](new_data)
