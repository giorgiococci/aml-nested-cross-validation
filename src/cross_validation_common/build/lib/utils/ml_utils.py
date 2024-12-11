import os
import json
import re

import pandas as pd

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.fsspec import AzureMachineLearningFileSystem

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

def get_mlflow_tracking_uri(ml_client):
    """Retrieves the MLflow tracking URI from the environment variable or from the ML client.

    Args:
        ml_client (MLClient): The machine learning client used to access the workspace.

    Returns:
        str: The MLflow tracking URI.
    """
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if mlflow_tracking_uri:
        return mlflow_tracking_uri
    
    return str(ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri)
    

def get_datastore_path(ml_client, datastore_name):
    
    subscription_id = ml_client.subscription_id
    resource_group = ml_client.resource_group_name
    workspace_name = ml_client.workspace_name
    
    path = "azureml://subscriptions/{}/".format(subscription_id)
    path += "resourcegroups/{}/".format(resource_group)
    path += "workspaces/{}/datastores/{}/paths/".format(workspace_name, datastore_name)

    return path

def get_datastore_files(ml_client, datastore_name, search_pattern: str):
    
    datastore_path = get_datastore_path(ml_client, datastore_name)
    
    fs = AzureMachineLearningFileSystem(datastore_path)
    file_list = fs.glob('**/*')
    
    files = [filename for filename in file_list if re.match(search_pattern, filename)]
    
    return files

def register_dataset(ml_client, dataset_name, dataset_version, dataset_type, path=None, description=None, tags=None):
    
    target_dataset_type = AssetTypes.URI_FILE
    
    target_dataset = Data(
        name=dataset_name,
        version=str(dataset_version),
        description=description,
        tags=tags,
        path=path,
        type=target_dataset_type,
    )
    
    return ml_client.data.create_or_update(target_dataset)

def download_datastore_file(ml_client, datastore_name, uri_file, local_path):
    datastore_path = get_datastore_path(ml_client, datastore_name)
    
    fs = AzureMachineLearningFileSystem(datastore_path)
    
    fs.get(rpath=uri_file, lpath=local_path)
    
    return local_path

def convert_urifolder_dataset_to_dataframe(input_folder, extension=".csv", **kwargs):
    output_dataframe = []

    # Walk through the input folder and find all CSV files
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(extension):
                csv_file_path = os.path.join(root, file)
                
                # Read each CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file_path, **kwargs)
                output_dataframe.append(df)

                print(f"Loaded {file}, shape: {df.shape}")

    # Concatenate all the DataFrames into one
    if output_dataframe:
        combined_df = pd.concat(output_dataframe, ignore_index=True)
    else:
        print(f"No {extension} files found in the directory.")
        
    return combined_df