from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo, upload_file
import os

# Ensure the hf_token is available (it's set in DEoKvud1Ze4d)
# You might need to retrieve it from an environment variable if running outside Colab's session
# For simplicity, we'll assume hf_token is accessible or hardcoded for now, as it is in DEoKvud1Ze4d
# In a real pipeline, always use os.getenv("HF_TOKEN")

# NOTE: For local execution, ensure hf_token is defined or uncomment the os.getenv line with your token.
hf_token = "HF_TOKEN" # Using the token from DEoKvud1Ze4d for consistency

repo_id = "abhishek1504/wellness-tourism-dataset" # Aligning with how data is loaded later
repo_type = "dataset"

# Initialize API client
api = HfApi(token=hf_token)

# Define the local path to tourism.csv
local_csv_path = "tourism_project/data/tourism.csv"

# Check if the local file exists
if not os.path.exists(local_csv_path):
    print(f"Error: The file {local_csv_path} does not exist. Please upload tourism.csv to tourism_project/data/.")
else:
    # Step 1: Check if the space exists, or create it
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=hf_token)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
        print(f"Space '{repo_id}' created.")
    except HfHubHTTPError as e:
        print(f"An HTTP error occurred: {e}. Check your HF_TOKEN and repository ID.")
        print("Make sure your token has 'write' access if you're trying to create a repo or upload files.")
    except Exception as e:
        print(f"An unexpected error occurred during repo check/creation: {e}")

    # Upload the tourism.csv file
    try:
        upload_file(
            path_or_fileobj=local_csv_path,
            path_in_repo="tourism.csv", # The name the file will have in the repo
            repo_id=repo_id,
            repo_type=repo_type,
            token=hf_token,
        )
        print(f"Successfully uploaded {local_csv_path} to {repo_id} as tourism.csv.")
    except HfHubHTTPError as e:
        print(f"An HTTP error occurred during file upload: {e}. Check your HF_TOKEN and permissions.")
    except Exception as e:
        print(f"An unexpected error occurred during file upload: {e}")
