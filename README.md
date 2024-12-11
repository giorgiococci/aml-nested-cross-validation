# Azure Login and Set Subscription

```sh
# Log in to Azure with a specific tenant
az login --tenant 16b3c013-d300-468d-ac64-7eda0820b6d3

# Set the subscription
az account set --subscription 6e71006f-51e9-4cb5-a0d0-a2ada9895e67
```

```sh
# Set the resource group
az configure --defaults group=svision

# Set the Azure Machine Learning workspace
az configure --defaults workspace=gcsvisionml
```

## Create AML environment

```sh
az ml environment create --file data-science/environments/environment.yml
az ml environment create --file data-science/environments/outer-loop-tasks/environment.yml --build-context ../../../ --dockerfile-path Dockerfile
```

## Run AML pipeline

```sh
az ml job create --file pipeline.yml
```
