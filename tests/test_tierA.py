import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import allure
import time

# Import necessary functions/fixtures from conftest.py
# We import the fixture function (tierA_scenarios) and the builder function
from tests.conftest import load_tierA_scenarios, create_deepeval_test_case 

# --- GEval Criteria Definition ---
# Note: The actual GEval object will be defined *inside* the test function 
# so it can access the 'azure_model' fixture.

TIERA_DATA = load_tierA_scenarios()

# Use pytest.mark.parametrize to run the test function for every scenario
@pytest.mark.parametrize(
    "scenario_data", 
    # CRITICAL: Call the fixture function (tierA_scenarios()) to get the list of data
    TIERA_DATA, 
    # Use the scenario_name for clear output in the test report
    ids=[s["scenario_name"] for s in TIERA_DATA]
)
def test_all_tierA_scenarios(azure_model, scenario_data):

    #Define G-Eval Metrics

    Correctness = GEval(
        name="Correctness Evaluation",
        evaluation_steps=[
        "1. Read the 'actual output' and compare the findings to what is mentioned in the 'expected output'"
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=azure_model,
        threshold=0.8
    )


    test_case = create_deepeval_test_case(scenario_data)

    metrics_to_run = [
        
        Correctness
    ]
    
    
    
    #ALLURE REPORTING
     # --- 2. RUN METRICS INDIVIDUALLY AND COLLECT RESULTS ---
    # We will use a dictionary to store results and track failure states
    results = {}
    test_failed = False
    
    # Use a try/except structure for each metric to prevent the entire test from exiting 
    # prematurely before other metric scores are calculated/logged.
    for metric in metrics_to_run:
        try:
            # Calculate the score and reason *without* using assert_test yet
            metric.measure(test_case) 
            
            results[metric.name] = {
                "score": metric.score,
                "threshold": metric.threshold,
                "reason": metric.reason,
                "status": "PASS" if metric.is_successful() else "FAIL"
            }
            if not metric.is_successful():
                test_failed = True
                
        except Exception as e:
            # Handle unexpected errors during metric evaluation (e.g., LLM server error)
            results[metric.name] = {
                "score": 0.0,
                "threshold": metric.threshold,
                "reason": f"Evaluation Error: {e}",
                "status": "ERROR"
            }
            test_failed = True

     #**********NEW ALLURE REPORTING **********
    with allure.step(f"Scenario Evaluation: {scenario_data['scenario_name']}"):
    
    # --- Attach Input Data ---
    # Attach the input data (assumed to be available as test_case.input)
    # Using JSON attachment type is good if the input is structured data
        allure.attach(
            test_case.input,
            name="Input Data",
            attachment_type=allure.attachment_type.JSON
        )

    # --- Attach Output Data ---
    # Attach the raw AI output (assumed to be available as test_case.actual_output)
        allure.attach(
            test_case.actual_output,
            name="Output Data",
            attachment_type=allure.attachment_type.JSON
        )

    # --- Log Metric Details (Including Threshold) ---
    for name, data in results.items():
        
        # 2. Format the detailed metric output as a single text block
        #    Note: We include the threshold here
        metric_details = (
            f"Metric: {name}\n"
            f"Status: {data['status']}\n"
            f"Score: {data['score']:.4f}\n"
            f"Threshold: {data['threshold']:.4f}\n" # <--- THRESHOLD INCLUDED HERE
            f"Reasoning: {data['reason']}"
        )
        
        # 3. Attach the detailed block as a TEXT attachment
        allure.attach(
            metric_details,
            name=f"Metric Result: {name} ({data['status']})",
            attachment_type=allure.attachment_type.TEXT
        )


    # --- 4. FINAL ASSERTION ---
    # This single, final assertion controls the overall test status in Pytest/Allure.
    assert test_failed is False, "One or more DeepEval metrics failed. Check attached report details."
    time.sleep(40)  # Wait 40 seconds between tests to avoid hitting rate limits
    