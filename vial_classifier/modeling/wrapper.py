"""MLflow pyfunc serving wrapper.                             Owner: Person B

To be extracted from notebooks/01_training.ipynb, cell 35: SimpleLoggingWrapper.

Contract (used by scripts/validate_model.py and POST /invocations):
    predict({"instances": [<base64 image>, ...]})
      -> {"predictions": [...], "confidences": [...], "raw_outputs": [...], "num_samples": n}
Every request is appended to config.INFERENCE_LOG_PATH for drift detection.
"""
