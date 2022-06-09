# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Lightweight component tuning function."""
from typing import NamedTuple

from kfp.v2.dsl import component

@component(
    base_image="python:3.8",
    output_component_file="beer_kfp_tune_hyperparameters.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)
def tune_hyperparameters(
    project: str,
    location: str,
    container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    staging_bucket: str,
    max_trial_count: int,
    parallel_trial_count: int,
) -> NamedTuple(
    "Outputs",
    [("best_map_at_10", float), ("best_factors", int), ("best_iterations", int), ("best_regularization", float)],
):

    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-16",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
            },
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name="beer_kfp_trial_job",
        worker_pool_specs=worker_pool_specs,
    )

    hp_job = aiplatform.HyperparameterTuningJob(
        display_name="beer_kfp_tuning_job",
        custom_job=custom_job,
        metric_spec={
            "map_at_10": "maximize",
        },
        parameter_spec={
            "regularization": hpt.DoubleParameterSpec(
                min=1.0e-4, max=1.0e-1, scale="log"
            ),
            "factors": hpt.DiscreteParameterSpec(
                values=[16, 32, 64], scale="linear"
            ),
            "iterations": hpt.IntegerParameterSpec(
                min=10, max=100, scale="linear"
            ),
        },
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )

    hp_job.run()

    metrics = [
        trial.final_measurement.metrics[0].value for trial in hp_job.trials
    ]
    best_trial = hp_job.trials[metrics.index(max(metrics))]
    best_map_at_10 = float(best_trial.final_measurement.metrics[0].value)
    best_factors = int(best_trial.parameters[0].value)
    best_iterations = int(best_trial.parameters[1].value)
    best_regularization = best_trial.parameters[2].value
    return best_map_at_10, best_factors, best_iterations, best_regularization
