def CTGAN_generator(loader, seed=0, list_ids=[], hparams={}):
    """
    Generates synthetic data using a CTGAN model based on a specified dataset.

    Args:
        pb (str):
            The name of the dataset to be used, referenced from `dict_loader`.
        seed (int, optional):
            A random seed for reproducibility. Defaults to 0.
        list_ids (list, optional):
            A list of specific IDs from the dataset to use. Defaults to an empty list.
        hparams (dict, optional):
            A dictionary of hyperparameters for potential future customization.
            Currently unused in this implementation. Defaults to an empty dictionary.

    Returns:
        pd.DataFrame:
            A synthetic dataset generated by the CTGAN model, with the same size and
            metadata structure as the original dataset.
    """
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    # Load the dataset using the `dict_loader` utility
    data = loader(object_bool=True, list_ids=list_ids).copy()

    # Detect metadata structure from the loaded dataset
    metadata_object = SingleTableMetadata()
    metadata_object.detect_from_dataframe(data=data)

    # Initialize and configure the CTGAN synthesizer
    synthesizer = CTGANSynthesizer(metadata_object)
    synthesizer.fit(data)

    # Generate and return synthetic data with the same size as the original dataset
    return synthesizer.sample(len(data))
