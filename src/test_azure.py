
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AzureOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv
import json
import requests
import time

#load environment variables
load_dotenv()

#read from endpoint instead of file
API_ENDPOINT = "https://localhost:7083/api/Proposals/test-triage" 


def get_ai_output_from_api(input_data: dict, output_path: str) -> str:
    """
    Calls the AI endpoint with a retry mechanism for rate limiting (HTTP 429).
    """
    
    # 1. Serialization of Input Data
    try:
        input_string = json.dumps(input_data, ensure_ascii=False)
    except TypeError as e:
        return f'{{"error": "Input Serialization Failed", "details": "{e}", "recommendation": "Decline"}}'

    # --- RETRY LOGIC PARAMETERS ---
    max_retries = 5
    base_wait_seconds = 15  # Start wait time based on the error message

    for attempt in range(max_retries):
        try:
            # 2. API Request
            resp = requests.post(
                API_ENDPOINT,
                data=input_string,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                verify=False,
                timeout=120,
            )
            
            # 3. Check for 429 (Too Many Requests) *before* raising for other statuses
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    # Calculate wait time: 3s, then 6s, then 12s (simple exponential backoff)
                    wait_time = base_wait_seconds * (2 ** attempt) 
                    print(f"\nRate limit hit (429). Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue  # Go to the next loop iteration (retry)
                else:
                    # Max retries reached, let the HTTPError handler catch the 429
                    resp.raise_for_status() 

            # 4. Check for all other HTTP errors (4xx or 5xx)
            resp.raise_for_status() 
            
            # 5. Success: Deserialize and Format Output
            api_data = resp.json()
            output_string = json.dumps(api_data, ensure_ascii=False, indent=4)

            # Write to output file (if needed for debugging)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_string)
            
            return output_string # Success! Exit the function

        except requests.exceptions.HTTPError as e:
            # Catches all non-429 HTTP errors on the last attempt
            status_code = e.response.status_code
            error_details = e.response.text if e.response.text else str(e)
            print(f"\n[API HTTP Error {status_code}] Final Failure.")
            return f'{{"error": "API HTTP Error", "status_code": {status_code}, "details": "{error_details.replace("\"", "")}", "recommendation": "Review Required"}}'
        
        except requests.exceptions.RequestException as e:
            # Handles network/connection errors (Timeout, DNS failure, etc.)
            print(f"\n[API Connection Error] Final Failure: {e.__class__.__name__}")
            return '{"error": "API Connection Failed", "recommendation": "Decline"}'
        
        except json.JSONDecodeError:
            print("\n[API Error] Invalid JSON Response from API. Final Failure.")
            return '{"error": "Invalid JSON Response", "recommendation": "Review Required"}'
            
    # Should be unreachable if max_retries > 0, but included for completeness
    return '{"error": "Exceeded Max Retries", "recommendation": "Decline"}'


#Read from retrival context text documents and combine into a single string

def get_retrieval_contexts():
    context_files = ["testdata/findings.txt", "testdata/triageservice.txt", "testdata/policysearch.txt"]
    all_contexts = []
    for filename in context_files:
        try:
            with open(filename, "r") as f:
                all_contexts.append(f.read())
        except FileNotFoundError:
            print(f"Error: Context file '{filename}' not found.")
            # Handle error appropriately (e.g., skip or fail)
    return all_contexts



#---- Deepeval Model Definition ------

# Define a custom class to wrap the Azure OpenAI client for DeepEval
class AzureOpenAIModel(DeepEvalBaseLLM):
    def __init__(self, api_key: str, endpoint: str, api_version: str, deployment_name: str):
        # Synchronous client for 'generate' method
        self.sync_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        # Asynchronous client for 'a_generate' method
        self.async_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        self.deployment_name = deployment_name

    def load_model(self):
        return self.sync_client

    def get_model_name(self):
        return self.deployment_name

    def generate(self, prompt: str) -> str:
        client = self.sync_client
        response = client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        client = self.async_client
        response = await client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

