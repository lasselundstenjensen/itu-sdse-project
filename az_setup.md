This guide outlines a step by step process for hosting the project on azure once you have a working docker image on ghcr.io
ANYTHING MARKED WITH <text> needs to be replaced by your own text. 

Replace old DO IaC
```console
mkdir -p ./infra/azure
cat > ./infra/azure/app.yml << 'EOF'
# Azure Container Apps spec: the equivalent of the DigitalOcean App Platform
# "app spec". It runs the model-serving Docker image from ghcr.io. Your
# pipeline builds and pushes the image; Azure only pulls and runs it.
#
# One-time setup: make the GHCR package public (Package settings -> Change
# visibility) so Azure can pull it without credentials. If you keep it private,
# add a `registries` entry plus a secret (see the note below the spec).
#
# Placeholders to fill in from your pipeline / Azure setup:
#   __OWNER__            GitHub user or org, lowercased (ghcr.io/__OWNER__/model-serve)
#   __TAG__              exact image tag your pipeline pushed (e.g. commit SHA)
#   __SUBSCRIPTION_ID__  your Azure subscription ID
#   __RESOURCE_GROUP__   resource group holding the Container Apps environment
#   __ENVIRONMENT__      name of the Container Apps environment
#
# Deploy with:
#   az containerapp create -g __RESOURCE_GROUP__ --yaml azure-app.yml
#   (use `az containerapp update ... --yaml` for later changes)

name: mlops-itu-jtk
type: Microsoft.App/containerApps
location: Norway East

properties:
  managedEnvironmentId: /subscriptions/__SUBSCRIPTION_ID__/resourceGroups/__RESOURCE_GROUP__/providers/Microsoft.App/managedEnvironments/__ENVIRONMENT__

  configuration:
    # There is no per-path ingress routing to configure: a Container App has a
    # single ingress that sends ALL paths (/invocations, /infer, /health,
    # /drift) to the container with the path intact, so the DO ingress rules
    # are no longer needed.
    ingress:
      external: true        # public URL
      targetPort: 5001      # must match the port uvicorn listens on
      transport: auto
      allowInsecure: false  # redirect HTTP -> HTTPS

  template:
    containers:
      - name: model-serve
        image: ghcr.io/__OWNER__/model-serve:__TAG__
        resources:
          # Consumption plan only allows fixed CPU/memory pairs; 1 vCPU
          # pairs with 2Gi. For 1Gi of RAM use cpu: 0.5 instead.
          cpu: 1.0
          memory: 2Gi
        probes:
          # Model loading is slow, so a startup probe holds off the other
          # probes until /health first succeeds (~30s + 10 x 10s = up to 130s).
          - type: Startup
            httpGet:
              path: /health
              port: 5001
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 10
            failureThreshold: 10
          # Restart the container if /health starts failing later on.
          - type: Liveness
            httpGet:
              path: /health
              port: 5001
            periodSeconds: 10
            timeoutSeconds: 10
            failureThreshold: 3

    scale:
      # Keep one replica always running so you don't pay a slow cold start
      # (model reload) after scaling to zero. Inference log files live in the
      # container and are lost on every redeploy, same as before.
      minReplicas: 1
      maxReplicas: 1"
EOF
```

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

