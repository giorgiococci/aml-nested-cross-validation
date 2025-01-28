# Azure Login and Set Subscription

```sh
# Log in to Azure with a specific tenant
az login

# Set the subscription
az account set --subscription <subscription-name>
```

```sh
# Set the resource group
az configure --defaults group=<resource group>

# Set the Azure Machine Learning workspace
az configure --defaults workspace=<workspace name>
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
