import os

from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.aiplatform import (
    EndpointCreateOp,
    ModelDeployOp,
    ModelUploadOp,
)
from google_cloud_pipeline_components.experimental import (
    hyperparameter_tuning_job,
)
from google_cloud_pipeline_components.experimental.custom_job import (
    CustomTrainingJobOp,
)
from kfp.v2 import dsl

PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")

TRAINING_FILE_PATH = os.getenv("TRAINING_FILE_PATH")
VALIDATION_FILE_PATH = os.getenv("VALIDATION_FILE_PATH")

MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))

PIPELINE_NAME = os.getenv("PIPELINE_NAME", "beer")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", PIPELINE_ROOT)
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", PIPELINE_NAME)

@dsl.pipeline(
    name=f"{PIPELINE_NAME}-kfp-pipeline",
    description="Kubeflow pipeline that tunes, trains, and deploys on Vertex",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINING_CONTAINER_IMAGE_URI,
            },
        }
    ]

    metric_spec = hyperparameter_tuning_job.serialize_metrics(
        {"map_at_10": "maximize"}
    )

    parameter_spec = hyperparameter_tuning_job.serialize_parameters(
        {
            "regularization": hpt.DoubleParameterSpec(
                min=1.0e-4, max=1.0e-1, scale="log"
            ),
            "factors": hpt.DiscreteParameterSpec(
                values=[16, 32, 64], scale="linear"
            ),
            "iterations": hpt.IntegerParameterSpec(
                min=10, max=100, scale="linear"
            ),
        }
    )

    hp_tuning_task = hyperparameter_tuning_job.HyperparameterTuningJobRunOp(
        display_name=f"{PIPELINE_NAME}-kfp-tuning-job",
        project=PROJECT_ID,
        location=REGION,
        worker_pool_specs=worker_pool_specs,
        study_spec_metrics=metric_spec,
        study_spec_parameters=parameter_spec,
        max_trial_count=MAX_TRIAL_COUNT,
        parallel_trial_count=PARALLEL_TRIAL_COUNT,
        base_output_directory=PIPELINE_ROOT,
    )

    trials_task = hyperparameter_tuning_job.GetTrialsOp(
        gcp_resources=hp_tuning_task.outputs["gcp_resources"]
    )

    best_hyperparameters_task = (
        hyperparameter_tuning_job.GetBestHyperparametersOp(
            trials=trials_task.output, study_spec_metrics=metric_spec
        )
    )

    # Construct new worker_pool_specs and
    # train new model based on best hyperparameters
    worker_pool_specs_task = hyperparameter_tuning_job.GetWorkerPoolSpecsOp(
        best_hyperparameters=best_hyperparameters_task.output,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-16"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": TRAINING_CONTAINER_IMAGE_URI,
                },
            }
        ],
    )

    training_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=f"{PIPELINE_NAME}-kfp-training-job",
        worker_pool_specs=worker_pool_specs_task.output,
        base_output_directory=BASE_OUTPUT_DIR,
    )

    model_upload_task = ModelUploadOp(
        project=PROJECT_ID,
        display_name=f"{PIPELINE_NAME}-kfp-model-upload-job",
        artifact_uri=f"{BASE_OUTPUT_DIR}/model",
    )
    model_upload_task.after(training_task)

    endpoint_create_task = EndpointCreateOp(
        project=PROJECT_ID,
        display_name=f"{PIPELINE_NAME}-kfp-create-endpoint-job",
    )
    endpoint_create_task.after(model_upload_task)

    model_deploy_op = ModelDeployOp(  # pylint: disable=unused-variable
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"],
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )
