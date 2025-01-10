from pathlib import Path
import os
import time

from avatars.models import AnalysisStatus
from avatars.client import ApiClient
from avatars.models import (
    AvatarizationJobCreate,
    AvatarizationParameters,
    ImputationParameters,
    ImputeMethod,
)
from avatars.processors import ToCategoricalProcessor

from utils.data_loader import func_loader

os.environ["AVATAR_BASE_URL"] = AVATAR_BASE_URL
os.environ["AVATAR_USERNAME"] = AVATAR_USERNAME
os.environ["AVATAR_PASSWORD"] = AVATAR_PASSWORD
client = ApiClient(base_url=os.environ.get("AVATAR_BASE_URL"))
client.authenticate(
    username=os.environ.get("AVATAR_USERNAME"),
    password=os.environ.get("AVATAR_PASSWORD"),
)


algo = "Avatar"
for pb in ["AIDS"]:
    for seed in range(1):
        df = func_loader[pb]()

        processor = ToCategoricalProcessor(to_categorical_threshold=20)
        processed = processor.preprocess(df)

        dataset = client.pandas_integration.upload_dataframe(processed)

        dataset = client.datasets.analyze_dataset(dataset.id)

        while dataset.analysis_status != AnalysisStatus.done:
            dataset = client.datasets.get_dataset(dataset.id)
            time.sleep(1)

        print(
            f"Lines: {dataset.nb_lines}, dimensions: {dataset.nb_dimensions}"
        )

        imputation = ImputationParameters(
            method="mode", k=8, training_fraction=0.3
        )

        job = client.jobs.create_avatarization_job(
            AvatarizationJobCreate(
                parameters=AvatarizationParameters(
                    k=20,
                    dataset_id=dataset.id,
                    imputation=ImputationParameters(method=ImputeMethod.mode),
                    seed=seed,
                    ncp=5,
                ),
            )
        )

        job = client.jobs.get_avatarization_job(id=job.id, timeout=1000)
        print(job.status)
        while job.status.value != "success":
            job = client.jobs.get_avatarization_job(id=job.id, timeout=1000)
            print("waiting", job.status)
            time.sleep(5)
        print(job.result)

        # Download the avatars as a pandas dataframe
        avatars_categorical = client.pandas_integration.download_dataframe(
            job.result.sensitive_unshuffled_avatars_datasets.id
        )

        avatars_categorical = processor.postprocess(df, avatars_categorical)
        print(avatars_categorical.head())
        avatars_categorical.to_csv(
            Path("Data", algo, f"{algo}_{pb}_seed{seed}.csv"), index=False
        )

        print(pb, seed)
        print("waiting")
        time.sleep(5)
