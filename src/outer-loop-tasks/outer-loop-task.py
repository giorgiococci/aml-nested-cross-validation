import argparse
import mlflow
import json
import os
import shutil
# from azureml.core import Workspace, ScriptRunConfig, Environment
# from azureml.train.hyperdrive import RandomParameterSampling, PrimaryMetricGoal, HyperDriveConfig, choice

from azure.ai.ml import MLClient
from azure.ai.ml import load_job
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

def create_ml_client(credential, subscription_id, resource_group, workspace_name):
    if not (subscription_id and resource_group and workspace_name):
        # Try to retrieve workspace details from environment variables
        subscription_id = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = resource_group or os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = workspace_name or os.getenv("AZURE_WORKSPACE_NAME")

    if subscription_id and resource_group and workspace_name:
        return MLClient(credential, subscription_id, resource_group, workspace_name)
    else:
        raise ValueError("Workspace details are incomplete.")

def get_ml_client(subscription_id=None, resource_group=None, workspace_name=None, config_path="config.json"):
    # Attempt to authenticate using DefaultAzureCredential
    try:
        credential = DefaultAzureCredential()
        ml_client = create_ml_client(credential, subscription_id, resource_group, workspace_name)
        print("Authenticated using DefaultAzureCredential.")
        return ml_client
    except Exception as e:
        print(f"DefaultAzureCredential authentication failed: {e}")

    # Attempt to authenticate using InteractiveBrowserCredential
    try:
        credential = InteractiveBrowserCredential()
        ml_client = create_ml_client(credential, subscription_id, resource_group, workspace_name)
        print("Authenticated using InteractiveBrowserCredential.")
        return ml_client
    except Exception as e:
        print(f"InteractiveBrowserCredential authentication failed: {e}")


    # Fallback to configuration file
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
            subscription_id = subscription_id or config.get("subscription_id")
            resource_group = resource_group or config.get("resource_group")
            workspace_name = workspace_name or config.get("workspace_name")

        credential = DefaultAzureCredential()  # Re-attempt using DefaultAzureCredential for consistency
        ml_client = create_ml_client(credential, subscription_id, resource_group, workspace_name)
        print("Authenticated using DefaultAzureCredential with config file details.")
        return ml_client
    except Exception as e:
        print(f"Fallback to config file failed: {e}")
        
    # Try to retrieve workspace details from mlflow environment variables
    try:
        uri = os.environ["MLFLOW_TRACKING_URI"]
        
        print(f"Retrieved workspace details from MLFLOW_TRACKING_URI: {uri}")
        
        uri_segments = uri.split("/")
        subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
        resource_group_name = uri_segments[uri_segments.index("resourceGroups") + 1]
        workspace_name = uri_segments[uri_segments.index("workspaces") + 1]
        
        credential = DefaultAzureCredential()
        ml_client = create_ml_client(credential, subscription_id, resource_group_name, workspace_name)
        
        print("Authenticated using AzureMLOnBehalfOfCredential. Retrieved workspace details from MLFLOW_TRACKING_URI.")
        print(f"Subscription ID: {subscription_id}, Resource Group: {resource_group_name}, Workspace Name: {workspace_name}")
        
        return ml_client
    except Exception as e:
        print(f"Failed to retrieve workspace details from mlflow environment variables: {e}")

    raise Exception("Failed to authenticate and retrieve MLClient.")

def init():
    print("Outer Loop Task Initialized!")
    
    global dataset_path, sweep_pipeline_path

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="ParallelRunStep Agent"
    )
    parser.add_argument("--dataset_input", type=str)
    parser.add_argument("--sweep_pipeline_path", type=str)
    
    args, _ = parser.parse_known_args()

    dataset_path = args.dataset_input
    sweep_pipeline_path = args.sweep_pipeline_path
    
def run(input_data):
    print(f"run method start: {__file__}, run({input_data})")
    
    for fold_file in input_data:
        # Read load fold file
        with open(fold_file, "r") as f:
            fold = json.load(f)
        
        # Create train-test splits for outer cross-validation folds
        train_indices = fold["train"]
        test_indices = fold["test"]

        print("Train indices for each fold:", train_indices)
        print("Test indices for each fold:", test_indices)
        import pandas as pd
        
        print(f"Dataset path: {dataset_path}")

        dataset = pd.read_csv(dataset_path)
        
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset columns: {dataset.columns}")
        print(f"Dataset head: {dataset.head()}")
        
        # Get train and test datasets from the fold indices
        train_dataset = dataset.iloc[train_indices]
        test_dataset = dataset.iloc[test_indices]
            
        print(f"Train Dataset shape: {train_dataset.shape}")
        print(f"Train Dataset columns: {train_dataset.columns}")
            
        # Convert train and test datasets to csv files
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        print(f"Output folder: {output_folder}")
            
        # Get fold file name
        fold_file_name = os.path.basename(fold_file)
        train_dataset_path = os.path.join(output_folder, fold_file_name.replace(".json", "_train.csv"))
        test_dataset_path = os.path.join(output_folder, fold_file_name.replace(".json", "_test.csv"))
        
        train_dataset.to_csv(train_dataset_path, index=False)
        test_dataset.to_csv(test_dataset_path, index=False)
        
        print(f"Train dataset saved to: {train_dataset_path}")
        print(f"Test dataset saved to: {test_dataset_path}")
        
        ml_client = get_ml_client()
        
        # Copy the sweep pipeline folder (and all its content) to the current directory
        shutil.copytree(sweep_pipeline_path, ".", dirs_exist_ok=True)
            
        # Get current run ID
        from azureml.core import Run
        run = Run.get_context()
        current_run_id = run.id
        
        pipeline_job = load_job(source="sweep-pipeline.yml")
        submitted_job = ml_client.jobs.create_or_update(
            job=pipeline_job, 
            experiment_name="hyperparameter-tuning",
            tags={"parent_run_id": current_run_id}
            )
        
        # Get submitted job id
        submitted_job_id = submitted_job.id
        
        # Add the submitted job id to the current run
        run.set_tags({"sweep_pipeline_job_id": submitted_job_id})

    return []