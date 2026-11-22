import pandas as pd
import base64
import requests
from pathlib import Path
from fastapi.testclient import TestClient
from src.deployment.fastapi import app

def test_endpoint():
    #load sample metada
    test_metadata = pd.read_csv("data/test/metadata.csv")
    sample_file = test_metadata.iloc[0]["filename"]
    sample_label = int(test_metadata.iloc[0]["class"])
    print(f"Sample image: {sample_file} (label={sample_label})")

    #read img and convert to base64
    with open(Path("data/test") / sample_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {"instances": [img_b64]}

    client = TestClient(app)
    r = client.post(f"/infer", json=payload)
    
    assert r.status_code == 200
    
    result = r.json()

    #check that predictions are returned
    assert result is not None

if __name__ == "__main__":
    test_endpoint()
