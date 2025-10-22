import pytest
import os
import json
from deepeval.test_case import LLMTestCase
from src.test_azure import AzureOpenAIModel, get_retrieval_contexts, get_ai_output_from_api

def load_manifest(filename: str) -> list[dict]:
    """Loads the list of scenario metadata from a manifest file."""
    path = os.path.join("testdata", filename)
    with open(path, "r") as f:
        return json.load(f)

# --- Fixture to load all Low Risk scenarios for parameterization ---

def load_low_risk_scenarios() -> list[dict]:
    """Provides the list of metadata for all low-risk test cases."""
    return load_manifest("low_risk/dataset_low_risk.json")

def load_bias_scenarios() -> list[dict]:
    """Provides the list of metadata for all bias test cases."""
    return load_manifest("bias/dataset_bias.json")

# --- Function to convert manifest item into an LLMTestCase ---

def create_deepeval_test_case(scenario: dict) -> LLMTestCase:
    """
    Takes a dictionary from the manifest, reads the input file,
    CALLS THE API FOR OUTPUT, and builds the full LLMTestCase.
    """
    
    # 1. DEFINE PATH VARIABLES
    root_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # Build the full, absolute paths to the input and output files
    # The '..' navigates up from 'tests/' to the project root.
    input_content_path = os.path.join(root_dir, "..", "testdata", scenario["input_file"])
    output_content_path = os.path.join(root_dir, "..", "testdata", scenario["output_file"])
    
    # 2. Read the Input Data (the payload for the API)
    # input_content_path is used here:
    with open(input_content_path, "r") as f:
        input_data = json.load(f)
        input_string = json.dumps(input_data, ensure_ascii=False, indent=4)
        
    # 3. Dynamic API call (uses input_data, not a path)
    actual_output_string = get_ai_output_from_api(input_data, output_content_path)
    
    # ... rest of the function remains the same ...
    
    return LLMTestCase(
        input=input_string,
        actual_output=actual_output_string,
        retrieval_context=get_retrieval_contexts()
        # ... meta data ...
    )

#--- Pytest Fixture for Model Initialization ---
@pytest.fixture(scope="session")
def azure_model():
    """
    Initializes and returns the AzureOpenAIModel instance.
    It will now read variables loaded from your .env file.
    """
    # Load Azure credentials from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([api_key, endpoint, api_version, deployment_name]):
        pytest.fail("Please set all required Azure OpenAI environment variables in your .env file.")

    # Initialize and return the custom model wrapper
    return AzureOpenAIModel(api_key, endpoint, api_version, deployment_name)

