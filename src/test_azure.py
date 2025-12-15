
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
#API_ENDPOINT = "https://localhost:7001/api/Proposals/test-triage"  



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
    max_retries = 30
    base_wait_seconds = 3  # Start wait time based on the error message

    for attempt in range(max_retries):
        try:
            # 2. API Request
            resp = requests.post(
                API_ENDPOINT,
                data=input_string,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                verify=False,
                timeout=600,
            )
            
            # 3. Check for 429 OR 400 with Rate Limit in body
            if resp.status_code == 429:
            # Existing logic for standard 429 (Too Many Requests)
                is_rate_limit = True
            
            elif resp.status_code == 400 and "RateLimitReached" in resp.text:
                # New check: 400 status, but the body contains the rate limit message
                is_rate_limit = True

                # Extract the suggested wait time from the body text
                import re
                match = re.search(r"Please retry after (\d+) seconds", resp.text)
                if match:
                    # Override the Retry-After header check below with this value
                    resp.headers['Retry-After'] = match.group(1)

            else:
                is_rate_limit = False

            if is_rate_limit:
                if attempt < max_retries - 1:

                    wait_time = None
                    # Check for the Retry-After header (or the value extracted above)
                    if 'Retry-After' in resp.headers:
                        try:
                            # Use the server-suggested time
                            wait_time = int(resp.headers['Retry-After'])
                            print(f"\nServer suggested wait time: {wait_time}s.")
                        except ValueError:
                            pass # Fall through to backoff if header value is bad

                    if wait_time is None:
                        # Fall back to your exponential backoff
                        wait_time = base_wait_seconds * (2 ** attempt) 
        
                    print(f"\nRate limit hit ({resp.status_code}). Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue # Go to the next loop iteration (retry)

                else:
                    # Max retries reached
                    resp.raise_for_status() # Let the HTTPError handler catch the 400
            

            # 4. Check for all other HTTP errors (4xx or 5xx)
            resp.raise_for_status() 
            
            # 5. Success: Deserialize and Format Output
            api_data = resp.json()
            output_string = json.dumps(api_data, ensure_ascii=False, indent=4)

            # Write to output file (if needed for debugging)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_string)

            print(f"Test successful. Applying global throttle")
            time.sleep(10)  # Short wait after success to avoid immediate rate limits
            
            return output_string # Success! Exit the function

        except requests.exceptions.HTTPError as e:
            # Catches all non-429 HTTP errors on the last attempt
            status_code = e.response.status_code
            error_details = e.response.text if e.response.text else str(e)
            print(f"\n[API HTTP Error {status_code}] Final Failure.")
            return f'{{"error": "API HTTP Error", "status_code": {status_code}, "details": "{error_details.replace("\"", "")}", "recommendation": "Review Required"}}'
        

        # Catches CONNECTION Errors (Timeouts, DNS, etc.) and RETRY
        # -------------------------------------------------------------
        except requests.exceptions.RequestException as e:
            # Check if we have attempts remaining
            if attempt < max_retries - 1:
                
                # Use exponential backoff for connection errors
                wait_time = base_wait_seconds * (2 ** attempt) 
                
                print(f"\n[API Connection Error: {e.__class__.__name__}]. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
                continue  # CRITICAL: This sends execution back to the start of the loop
            else:
                # Max retries reached, report final failure
                print(f"\n[API Connection Error] Final Failure: {e.__class__.__name__}")
                return '{"error": "API Connection Failed", "recommendation": "Decline"}'
        

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
    context_files = ["testdata/findings.txt", "testdata/triageDossier.txt", "testdata/policysearch.txt"]
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
    def __init__(self, api_key: str, endpoint: str, api_version: str, deployment_name: str, temperature: float):
        # Synchronous client for 'generate' method
        self.temperature = temperature
        self.sync_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        # Asynchronous client for 'a_generate' method
        self.temperature = temperature
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
        #send temp
        print(f"DEBUG: Temperature being used in API call: {self.temperature}")
        response = client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        client = self.async_client
        response = await client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

