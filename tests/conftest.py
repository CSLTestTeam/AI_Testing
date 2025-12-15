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

def load_tierA_scenarios() -> list[dict]:
    """Provides the list of metadata for all low-risk test cases."""
    return load_manifest("tierA/dataset_tierA.json")

def load_bias_scenarios() -> list[dict]:
    """Provides the list of metadata for all bias test cases."""
    return load_manifest("bias/dataset_bias.json")

def load_tierB_scenarios() -> list[dict]:
    """Provides the list of metadata for all high risk test cases."""
    return load_manifest("tierB/dataset_tierB.json")

def load_tierC_scenarios() -> list[dict]:
    """Provides the list of metadata for all high risk test cases."""
    return load_manifest("tierC/dataset_tierC.json")

def load_mismatches_scenarios() -> list[dict]:
    """Provides the list of metadata for all mismatches test cases."""
    return load_manifest("mismatches/dataset_mismatches.json")

def load_finances_scenarios() -> list[dict]:
    """Provides the list of metadata for all finances test cases."""
    return load_manifest("finances/dataset_finances.json")

def load_adherence_scenarios() -> list[dict]:
    """Provides the list of metadata for all prompt adherence test cases."""
    return load_manifest("prompt_adherence/dataset_adherence.json")

def load_incomplete_data_scenarios() -> list[dict]:
    """Provides the list of metadata for all incomplete data test cases."""
    return load_manifest("incomplete_data/dataset_incomplete.json")

def load_boundary_values_scenarios() -> list[dict]:
    """Provides the list of metadata for all boundary values test cases."""
    return load_manifest("boundary_values/dataset_boundary.json")


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
    expected_output_string = scenario["expected_output_prompt"]
    
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
        expected_output=expected_output_string,
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
    #temperature
    temperature = 1.0

     # Validate that all required variables are set
    
    if not all([api_key, endpoint, deployment_name]):
        pytest.fail("Please set all required Azure OpenAI environment variables in your .env file.")

    # Initialize and return the custom model wrapper
    return AzureOpenAIModel(api_key, endpoint, api_version, deployment_name, temperature)

