def Identity(loader, seed=None, list_ids=[], hparams={}):
    """
    Loads and returns a copy of a dataset without applying any transformations.

    This function retrieves the dataset associated with the specified name (`pb`)
    from the `dict_loader` utility. It uses optional parameters to control
    the data-loading behavior, such as selecting specific IDs.

    Args:
        pb (str):
            The name of the dataset to load, as defined in `dict_loader`.
        seed (int, optional):
            Not used in this function but kept for consistency with similar interfaces.
            Defaults to None.
        list_ids (list, optional):
            A list of specific IDs from the dataset to load. Defaults to an empty list.
        hparams (dict, optional):
            Hyperparameters for potential future use. Currently unused.
            Defaults to an empty dictionary.

    Returns:
        pd.DataFrame:
            A copy of the dataset loaded using `dict_loader`.
    """
    # Load the dataset from `dict_loader` using the specified name (`pb`)
    return loader(
        object_bool=False,  # Specify that the object should not be treated as a boolean
        list_ids=list_ids,  # Optionally filter rows based on `list_ids`
    ).copy()  # Return a copy of the dataset
