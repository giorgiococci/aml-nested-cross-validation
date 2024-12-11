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
        train_dataset = []
        test_dataset = []
        
        for train_idx, test_idx in zip(train_indices, test_indices):
            train_data = dataset.iloc[train_idx]
            test_data = dataset.iloc[test_idx]

            train_dataset.append(train_data)
            test_dataset.append(test_data)
            
        # Convert train and test datasets to csv files
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Get fold file name
        fold_file_name = os.path.basename(fold_file)
            
        train_dataset_path = os.path.join(output_folder, fold_file_name.replace(".json", "_train.csv"))
        test_dataset_path = os.path.join(output_folder, fold_file_name.replace(".json", "_test.csv"))
        
        pd.concat(train_dataset).to_csv(train_dataset_path, index=False)
        pd.concat(test_dataset).to_csv(test_dataset_path, index=False)
        
        print(f"Train dataset saved to: {train_dataset_path}")
        print(f"Test dataset saved to: {test_dataset_path}")
        
        ml_client = get_ml_client()
        
        # Print all files in the current directory
        print("Files in current directory:")
        for file in os.listdir("."):
            print(file)
        
        # Copy the sweep pipeline folders files to the current directory
        sweep_pipeline_files = os.listdir(sweep_pipeline_path)
        for file in sweep_pipeline_files:
            print(f"Copying {file} to current directory")
            shutil.copyfile(os.path.join(sweep_pipeline_path, file), file)
            
        # Get current MLFlow run ID
        current_run_id = mlflow.active_run().info.run_id
        
        pipeline_job = load_job(source="sweep-pipeline.yml")
        submitted_job = ml_client.jobs.create_or_update(
            job=pipeline_job, 
            experiment_name="hyperparameter-tuning"
            )

    return []
    
    

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_path", type=str, required=True)
#     parser.add_argument("--folds_path", type=str, required=True)
#     parser.add_argument("--output_path", type=str, required=True)
#     args = parser.parse_args()

#     # Load Workspace
#     # ws = Workspace.from_config()

#     # Load dataset and folds
#     with open(args.folds_path, "r") as f:
#         folds = json.load(f)
    
    
#     # fold_index = int(os.environ.get("AZUREML_RUN_TOKEN"))  # Assuming a unique identifier for parallel task
#     # train_idx, test_idx = folds[fold_index]

#     # # Define Sweep Job for Hyperparameter Tuning
#     # sweep_script = "./sweep_task.py"
#     # environment = Environment.get(workspace=ws, name="AzureML-sklearn-1.0-ubuntu20.04-py38")
#     # param_sampling = RandomParameterSampling({
#     #     "n_estimators": choice(50, 100, 150),
#     #     "max_depth": choice(10, 20, None)
#     # })

#     # sweep_config = HyperDriveConfig(
#     #     run_config=ScriptRunConfig(
#     #         source_directory="./scripts",
#     #         script=sweep_script,
#     #         arguments=[
#     #             "--train_idx", train_idx,
#     #             "--test_idx", test_idx,
#     #             "--dataset_path", args.dataset_path
#     #         ],
#     #         environment=environment,
#     #     ),
#     #     hyperparameter_sampling=param_sampling,
#     #     primary_metric_name="accuracy",
#     #     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
#     #     max_total_trials=20,
#     #     max_concurrent_trials=5
#     # )

#     # # Submit Sweep Job
#     # sweep_run = ws.experiments["nested_cv_experiment"].submit(sweep_config)
#     # sweep_run.wait_for_completion(show_output=True)

#     # # Save Results
#     # sweep_best_model = sweep_run.get_best_run_by_primary_metric()
#     # with open(os.path.join(args.output_path, f"fold_{fold_index}_results.json"), "w") as f:
#     #     f.write(sweep_best_model.get_metrics())
    
#     print("Outer Loop Task Completed!")

# if __name__ == "__main__":
#     main()
