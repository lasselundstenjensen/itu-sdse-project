"""Check a running serving endpoint (local or deployed).

Waits for the scoring server to come up, sends one of the test images to the endpoint
as base64 (the portable contract - the server can also take file paths, but those only
work when the file is on the server's machine), and prints /version, /health and /drift.

CLI: `python -m vial_classifier.serving.check_endpoint`
"""

import base64
from pathlib import Path
from pprint import pformat
import time

from loguru import logger
import pandas as pd
import requests
import typer

from vial_classifier import config

app = typer.Typer()


def wait_for_server(base_url, max_attempts=120):
    """Wait for the scoring server to come up (loading torch takes a while)."""
    for attempt in range(max_attempts):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            logger.info(f"[{attempt}] /health -> {r.status_code} {r.text.strip()}")
            if r.status_code == 200:
                logger.info("Model server is ready!")
                break
        except requests.RequestException:
            logger.info(f"[{attempt}] waiting for the API server...")
        time.sleep(2)


def send_sample_image(base_url):
    """Send the first test image to /infer as base64."""
    test_metadata = pd.read_csv(config.TEST_METADATA_PATH)
    sample_file = test_metadata.iloc[0]["filename"]
    sample_label = int(test_metadata.iloc[0]["class"])
    logger.info(f"Sample image: {sample_file} (label={sample_label})")

    with open(Path(config.TEST_DATA_DIR) / sample_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {"instances": [img_b64]}

    r = requests.post(f"{base_url}/infer", json=payload, timeout=60)
    logger.info(f"POST /infer -> {r.status_code}\n{pformat(r.json())}")


def show_version_and_health(base_url):
    logger.info(f"GET /version -> {requests.get(f'{base_url}/version').json()}")
    logger.info(f"GET /health  -> {requests.get(f'{base_url}/health').json()}")


def show_drift(base_url):
    """The endpoint runs the same drift check over the requests it has seen."""
    logger.info(f"GET /drift ->\n{pformat(requests.get(f'{base_url}/drift').json())}")


@app.command()
def main(base_url: str = f"http://localhost:{config.SERVING_PORT}"):
    """Check that the endpoint is up and answers /infer, /version, /health and /drift."""
    wait_for_server(base_url)
    send_sample_image(base_url)
    show_version_and_health(base_url)
    show_drift(base_url)


if __name__ == "__main__":
    app()
