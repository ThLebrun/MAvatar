from typing import List
import pandas as pd
import numpy as np

CATEGORICAL_DTYPES = ["object", "category", "boolean"]


def get_continuous_under_threshold(
    df: pd.DataFrame, *, threshold: int = 10
) -> List[str]:
    """Get continuous variable names with number of unique values under a threshold."""
    columns_to_check = df.select_dtypes(exclude=CATEGORICAL_DTYPES).columns
    return list(
        filter(lambda col: df[col].nunique() <= threshold, columns_to_check)
    )


class ToCategoricalProcessor:
    """Processor to model selected numeric variables as categorical variables.

    Arguments
    ---------
        to_categorical_threshold:
            threshold of the number of distinct value to consider
            a continuous variable as categorical.

    Keyword Arguments
    -----------------
        keep_continuous:
            if `True`, continuous variables will be kept and
        suffixed with `continuous_suffix`.
        continuous_suffix:
            suffix for the continuous variable created during preprocess.
        category:
            if `keep_continuous=True`, name of the new category, needed for some specific
            avatarization cases with the use of group_modalities processor

    Examples
    --------
    With `keep_continuous=False` it only convert the variable to object.
    By this you ensure to keep all values during the avatarization.

    >>> df = pd.DataFrame(
    ...    {
    ...        "variable_1": [1, 7, 7, 1],
    ...        "variable_2": [1, 2, 7, 1]
    ...        }
    ...    )
    >>> processor = ToCategoricalProcessor(to_categorical_threshold = 2)
    >>> processor.preprocess(df).dtypes
    variable_1    object
    variable_2     int64
    dtype: object
    >>> avatar = pd.DataFrame(
    ...    {
    ...        "variable_1": [2, 1, 4, 1],
    ...        "variable_2": [2, 1, 4, 1]
    ...        }
    ...    )
    >>> avatar["variable_1"] = avatar["variable_1"].astype('object')
    >>> avatar.dtypes
    variable_1    object
    variable_2     int64
    dtype: object
    >>> processor.postprocess(df, avatar).dtypes
    variable_1    int64
    variable_2    int64
    dtype: object

    With `keep_continuous=True`, you duplicate the variable and keep it as continuous.
    This can be useful for other uses.

    >>> df = pd.DataFrame(
    ...    {
    ...        "variable_1": [1, 7, 7, 1],
    ...        "variable_2": [1, 2, 7, 1]
    ...        }
    ...    )
    >>> processor = ToCategoricalProcessor(to_categorical_threshold=2, keep_continuous=True)
    >>> processor.preprocess(df).dtypes
    variable_1          object
    variable_2           int64
    variable_1__cont     int64
    dtype: object
    """

    def __init__(
        self,
        to_categorical_threshold: int,
        *,
        keep_continuous: bool = False,
        continuous_suffix: str = "__cont",
        category: str = "other",
    ):
        self.to_categorical_threshold = to_categorical_threshold
        self.keep_continuous = keep_continuous
        self.continuous_suffix = continuous_suffix
        self.category = category

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform numeric variables into categorical variables.

        Arguments
        ---------
            df: dataframe to transform

        Returns
        -------
            DataFrame: transformed dataframe
        """
        df = df.copy()
        self.variables = get_continuous_under_threshold(
            df, threshold=self.to_categorical_threshold
        )

        # Perform the transformation by creating a new column cont_suffix where
        # NAs are kept as NAs.
        if self.keep_continuous:
            self.continuous_variables = [
                x + self.continuous_suffix for x in self.variables
            ]
            df[self.continuous_variables] = df[self.variables]

        df[self.variables] = df[self.variables].astype("object")
        return df

    def postprocess(
        self, source: pd.DataFrame, dest: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform converted categorical variables back to numeric.

        Arguments
        ---------
            source: reference data frame
            dest: data frame to transform

        Returns
        -------
            DataFrame: transformed data frame
        """
        dest = dest.copy()

        # affect the continuous value to modality "category",
        # needed for some specific avatarization cases with the use of group_modalities processor
        if self.keep_continuous:
            dest[self.variables] = np.where(
                dest[self.variables] == self.category,
                dest[self.continuous_variables],
                dest[self.variables],
            )
            dest = dest.drop(columns=self.continuous_variables)

        # reassign the original data type
        dest[self.variables] = dest[self.variables].astype(
            source[self.variables].dtypes.to_dict()
        )

        return dest


def SAIPH_Compressor(loader, seed=None, list_ids=[], hparams={"nf": 5}):
    import saiph

    nf = hparams["nf"]
    # data_full = dict_loader[pb]["Dataset"](object_bool=False).copy()#att
    data = loader(object_bool=False, list_ids=list_ids).copy()
    np.random.seed(seed)
    if len(list_ids) < 1:
        list_ids = list(range(len(data)))
    data.set_index(pd.Index(list(range(len(list_ids)))), inplace=True)
    processor = ToCategoricalProcessor(to_categorical_threshold=20)
    data = processor.preprocess(data)
    # data_full = processor.preprocess(data_full)#att
    # _, model = saiph.fit_transform(data_full, nf=nf)#att
    # coord = saiph.transform(data, model)#att

    coord, model = saiph.fit_transform(data, nf=nf)

    # project back
    A = pd.DataFrame(coord.iloc[:, :nf], columns=coord.columns[:nf])
    A = saiph.inverse_transform(A, model)
    A.set_index(pd.Index(list_ids), inplace=True)
    return A
