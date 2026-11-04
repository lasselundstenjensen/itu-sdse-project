"""FastAPI serving app (runs in the Docker image on port 5001).  Owner: Person C

To be extracted from notebooks/02_deployment.ipynb, cells 17 and 20:
- MLflow scoring server as a subprocess on UPSTREAM_PORT
- GET  /health      -> {"status": "ok"} once the model is loaded
- POST /invocations -> predictions; must accept a raw binary image body
                       (image/jpeg, application/octet-stream) as well as JSON
- POST /infer       -> alias of /invocations
- GET  /version     -> model version + training timestamp from version.json
- GET  /drift       -> monitoring.drift.detect_drift over the inference log
"""
