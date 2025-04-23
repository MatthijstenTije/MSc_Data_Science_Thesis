# main.py

import subprocess
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("Notebooks/Phase_02/logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_name):
    logger.info(f"Running: {script_name}")
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_name} failed:\n{e.stderr}")

def main():
    logger.info("Starting full pipeline...")

    run_script("Notebooks/Phase_02/generate_sentences.py")
    run_script("Notebooks/Phase_02/convert_logs_to_csv.py")

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
