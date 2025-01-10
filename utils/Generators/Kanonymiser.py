import pandas as pd


def Kanon(loader, seed=0, list_ids=[], hparams={}):
    k = hparams["k"]
    df = loader(object_bool=False, list_ids=list_ids)

    categorical = set(
        tuple(
            df.select_dtypes(include=["category", "object"]).columns.tolist()
        )
    )

    feature_columns = list(df.columns)
    sensitive_column = hparams["target"]
    feature_columns.remove(sensitive_column)
    # Assuming categorical_columns is a list of your categorical column names
    categorical_columns = df.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()

    # Convert values in categorical columns to string
    df[categorical_columns] = df[categorical_columns].astype(str)

    # print(df.head())

    for name in categorical:
        df[name] = df[name].astype("category")

    def get_spans(df, partition, scale=None):
        spans = {}
        for column in df.columns:
            if column in categorical:
                span = len(df[column][partition].unique())
            else:
                span = (
                    df[column][partition].max() - df[column][partition].min()
                )
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    full_spans = get_spans(df, df.index)
    # print(full_spans)

    def split(df, partition, column):
        dfp = df[column][partition]
        if column in categorical:
            values = dfp.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)

    def is_k_anonymous(df, partition, sensitive_column, k=k):
        if len(partition) < k:
            return False
        return True

    def partition_dataset(
        df, feature_columns, sensitive_column, scale, is_valid
    ):
        finished_partitions = []
        partitions = [df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = get_spans(df[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = split(df, partition, column)
                if not is_valid(df, lp, sensitive_column) or not is_valid(
                    df, rp, sensitive_column
                ):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    finished_partitions = partition_dataset(
        df, feature_columns, sensitive_column, full_spans, is_k_anonymous
    )

    # print(len(finished_partitions))

    def build_indexes(df):
        indexes = {}
        for column in categorical:
            values = sorted(df[column].unique())
            indexes[column] = {
                x: y for x, y in zip(values, range(len(values)))
            }
        return indexes

    def get_coords(df, column, partition, indexes, offset=0.1):
        if column in categorical:
            sv = df[column][partition].sort_values()
            l, r = (
                indexes[column][sv[sv.index[0]]],
                indexes[column][sv[sv.index[-1]]] + 1.0,
            )
        else:
            sv = df[column][partition].sort_values()
            next_value = sv[sv.index[-1]]
            larger_values = df[df[column] > next_value][column]
            if len(larger_values) > 0:
                next_value = larger_values.min()
            l = sv[sv.index[0]]
            r = next_value
        l -= offset
        r += offset
        return l, r

    def get_partition_rects(
        df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]
    ):
        rects = []
        for partition in partitions:
            xl, xr = get_coords(
                df, column_x, partition, indexes, offset=offsets[0]
            )
            yl, yr = get_coords(
                df, column_y, partition, indexes, offset=offsets[1]
            )
            rects.append(((xl, yl), (xr, yr)))
        return rects

    def get_bounds(df, column, indexes, offset=1.0):
        if column in categorical:
            return 0 - offset, len(indexes[column]) + offset
        return df[column].min() - offset, df[column].max() + offset

    def agg_categorical_column(series):
        return [",".join(set(series))]

    def agg_numerical_column(series):
        return [series.mean()]

    def build_anonymized_dataset(
        df, partitions, feature_columns, sensitive_column, max_partitions=None
    ):
        aggregations = {}
        for column in feature_columns:
            if column in categorical:
                aggregations[column] = agg_categorical_column
            else:
                aggregations[column] = agg_numerical_column
        rows = []
        for i, partition in enumerate(partitions):
            if i % 100 == 1:
                print("Finished {} partitions...".format(i))
            if max_partitions is not None and i > max_partitions:
                break
            grouped_columns = df.loc[partition].agg(
                aggregations, squeeze=False
            )
            sensitive_counts = (
                df.loc[partition]
                .groupby(sensitive_column)
                .agg({sensitive_column: "count"})
            )
            # print(grouped_columns)
            values_ = grouped_columns.to_dict()
            values = {}
            for key_ in values_.keys():
                values[key_] = values_[key_][0]
                # if type(values_[key_][0])==type('str'):
                #     values[key_] = values_[key_][0].split(',')[0]
                # else:
                #     values[key_] = values_[key_][0]

            for sensitive_value, count in sensitive_counts[
                sensitive_column
            ].items():
                if count == 0:
                    continue
                values.update(
                    {
                        sensitive_column: sensitive_value,
                        "count": count,
                    }
                )
                rows.append(values.copy())
        df = pd.DataFrame(rows)
        count_column = df["count"]

        return (
            df.loc[df.index.repeat(count_column)]
            .reset_index(drop=True)
            .drop("count", axis=1)
        )

    return build_anonymized_dataset(
        df, finished_partitions, feature_columns, sensitive_column
    )
