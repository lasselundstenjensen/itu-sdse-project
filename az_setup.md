This guide outlines a step by step process for hosting the project on azure once you have a working docker image on ghcr.io
ANYTHING MARKED WITH <text> needs to be replaced by your own text. 

Install az cli tool
MacOs:
```console
brew install azure-cli
```

Linux/WSL:
```console
curl -fsSL 'https://azurecliprod.blob.core.windows.net/$root/deb_install.sh' | sudo bash
```


Login using browser
```console
az login
```

Add extension and register namespaces
```console
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Create RG in the right region
```console
RG=mlops-itu-rg
ENV=mlops-itu-env
LOC=norwayeast

az group create --name $RG --location $LOC
az containerapp env create --name $ENV --resource-group $RG --location $LOC
```

Substitute infra fields with az query output
```console
SUB=$(az account show --query id -o tsv)
OWNER=<your-github-user-lowercase>
TAG=<your-image-tag>

sed \
  -e "s/__OWNER__/$OWNER/" \
  -e "s/__TAG__/$TAG/" \
  -e "s/__SUBSCRIPTION_ID__/$SUB/" \
  -e "s/__RESOURCE_GROUP__/$RG/" \
  -e "s/__ENVIRONMENT__/$ENV/" \
  ./infra/azure/app.yml > /tmp/app.rendered.yml

az containerapp create \
  --name mlops-itu-jtk \
  --resource-group $RG \
  --yaml /tmp/app.rendered.yml  
```

If new image needs to be pushed here's the update command
```console
RG=mlops-itu-rg
OWNER=<your-github-user-lowercase>
NEW_TAG=<your-new-tag>   # e.g. the commit SHA

az containerapp update \
  --name mlops-itu-jtk \
  --resource-group $RG \
  --image ghcr.io/$OWNER/model-serve:$NEW_TAG
```

