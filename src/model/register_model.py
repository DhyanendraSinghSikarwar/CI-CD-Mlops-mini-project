# register model

import json
import mlflow
import logging
import os
import dagshub
from dotenv import load_dotenv
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "dhyanendra.manit"
repo_name = "CI-CD-Mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# mlflow.set_tracking_uri('https://dagshub.com/dhyanendra.manit/mlops-mini-project.mlflow')
# dagshub.init(repo_owner='dhyanendra.manit', repo_name='mlops-mini-project', mlflow=True)

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Tag the MLflow run as 'production' (DagsHub does not support Model Registry)."""
    try:
        tracking_uri = mlflow.get_tracking_uri()
        if 'dagshub.com' in tracking_uri:
            client = mlflow.tracking.MlflowClient()
            run_id = model_info['run_id']
            client.set_tag(run_id, "model_status", "production")
            logger.info(f"Tagged run {run_id} as production.")
            print(f"Tagged run {run_id} as production.")
            return
        # If not on DagsHub, you could add model registry logic here for other MLflow servers
        logger.info('Model Registry logic skipped (not implemented for non-DagsHub).')
    except Exception as e:
        logger.error('Error during model tagging: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()