import numpy as np


def MST(loader, seed=None, list_ids=[], hparams={"epsilon": 3.0}):
    """
    Generates synthetic data using the MST (Maximum Spanning Tree) algorithm
    with differential privacy guarantees.

    Args:
        pb (str):
            The name of the dataset to load from `dict_loader`.
        seed (int, optional):
            A random seed for reproducibility. Defaults to None.
        list_ids (list, optional):
            A list of specific IDs from the dataset to use. Defaults to an empty list.
        hparams (dict, optional):
            Hyperparameters for the MST synthesizer:
            - "epsilon" (float): Privacy budget for differential privacy. Defaults to 3.0.

    Returns:
        pd.DataFrame:
            A synthetic dataset of the same size as the input dataset.
    """
    from snsynth import Synthesizer

    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Load the dataset using the `dict_loader` utility
    data = loader(object_bool=False, list_ids=list_ids).copy()

    # Create an MST synthesizer with the specified epsilon value
    generator = Synthesizer.create(
        "mst", epsilon=hparams["epsilon"], verbose=True
    )

    # Fit the synthesizer to the data with a preprocessor epsilon value
    preprocessor_eps = np.around(hparams["epsilon"] / 3, decimals=2)
    generator.fit(data, preprocessor_eps=preprocessor_eps)

    # Generate synthetic data of the same size as the original dataset
    synth = generator.sample(len(data))

    return synth
