import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import allure
import time

# Import necessary functions/fixtures from conftest.py
# We import the fixture function (low_risk_scenarios) and the builder function
from tests.conftest import load_low_risk_scenarios, create_deepeval_test_case 

# --- GEval Criteria Definition ---
# Note: The actual GEval object will be defined *inside* the test function 
# so it can access the 'azure_model' fixture.

LOW_RISK_DATA = load_low_risk_scenarios()

# Use pytest.mark.parametrize to run the test function for every scenario
@pytest.mark.parametrize(
    "scenario_data", 
    # CRITICAL: Call the fixture function (low_risk_scenarios()) to get the list of data
    LOW_RISK_DATA, 
    # Use the scenario_name for clear output in the test report
    ids=[s["scenario_name"] for s in LOW_RISK_DATA]
)
def test_all_low_risk_scenarios(azure_model, scenario_data):

    scenario_name = scenario_data['scenario_name']
    
    
    # 1. Define the GEval metric, passing the necessary model fixture
    ###rename to "process adherence metric"
    overall_model_evaluation_metric = GEval(
        name="Analysis Evaluation",
        evaluation_steps=[
        "1. Ignore and do not penalise any mention of the absence or unavailability of a credit report or any uses of it in the reasonings or any flags as a critical issue as this is an expected mistake of the model. Any further consequences of this missing credit report that are highlighted should also not be penalised.",
        "2. Check if the 'actual output' follows the steps detailed in 'retrieval context",
        "3. Assess if the 'actual output' covers ALL points in the 'retrieval context'.",
        "4. Mention where 'actual output' has failed to follow the 'retrieval context",
        "5. Mention and penalise where false mismatches have been highligted in the 'actual output'",
        "6. Once again, IGNORE and DO NOT penalise any mention of the absence or unavailability of a credit report or any uses of it in the reasonings",
        "7. IGNORE AND DO NOT PENALISE any flags of a missing credit report as a critical issue as this is an expected mistake of the model. Any further consequences of this missing credit report that are highlighted should also not be penalised.",
        "8. Assign a final score from 0.0 to 1.0 based on the combined assessment.",
        "9. Present findings in basic, easy to understand English"

        
    ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        model=azure_model,
        threshold=0.5
    )

    output_hallucination = GEval(
        name="Output Hallucination",
        evaluation_steps=[
        "1. Ignore and do not penalise any mention of the absence or unavailability of a credit report or any uses of it in the reasonings or any flags as a critical issue as this is an expected mistake of the model. Any further consequences of this missing credit report that are highlighted should also not be penalised.",
        "2. Read the 'actual output' and compare the findings to what is in the 'input'",
        "3. Assess if the 'actual output' is faithful. Mention any hallucinations or if anything has been made up except for the missing credit report",
        "4. Do not judge the 'actual_output' based on how well it follows the steps in the 'retrieval context'. The only thing you should judge is if any hallucinations are present except for the missing credit report",
        "5. Once again, IGNORE and DO NOT penalise any mention of the absence or unavailability of a credit report or any uses of it in the reasonings",
        "6. IGNORE AND DO NOT PENALISE any flags of a missing credit report as a critical issue as this is an expected mistake of the model. Any further consequences of this missing credit report that are highlighted should also not be penalised.",
        "7. Assign a final score from 0.0 to 1.0 based on the combined assessment of steps 1, 2, 3, and 4. **ENSURE the penalty for the continued mention and analysis based on a missing credit report (Step 1, 5 and 6) is 0.0.**",
        "8. Present findings in basic, easy to understand English"


        
    ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=azure_model,
        threshold=0.5
    )


    test_case = create_deepeval_test_case(scenario_data)

    metrics_to_run = [
        overall_model_evaluation_metric,
        output_hallucination
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
                "reason": metric.reason,
                "status": "PASS" if metric.is_successful() else "FAIL"
            }
            if not metric.is_successful():
                test_failed = True
                
        except Exception as e:
            # Handle unexpected errors during metric evaluation (e.g., LLM server error)
            results[metric.name] = {
                "score": 0.0,
                "reason": f"Evaluation Error: {e}",
                "status": "ERROR"
            }
            test_failed = True

    # --- 3. LOG ALL RESULTS TO ALLURE ---
    #log input
    allure.attach(
        test_case.input,
        name=f"Input Data: {scenario_data['scenario_name']}", # <-- Unique name defined once
        attachment_type=allure.attachment_type.JSON
    )
    #log output
    with allure.step(f"Evaluate Scenario: {scenario_data['scenario_name']}"):
        allure.attach(test_case.actual_output, name="AI Raw JSON Output", attachment_type=allure.attachment_type.JSON)

        for name, data in results.items():
            allure.attach(
                f"Score: {data['score']:.4f}\nStatus: {data['status']}\nReasoning: {data['reason']}",
                name=f"{name} ({data['status']})",
                attachment_type=allure.attachment_type.TEXT
            )

    # --- 4. FINAL ASSERTION ---
    # This single, final assertion controls the overall test status in Pytest/Allure.
    assert test_failed is False, "One or more DeepEval metrics failed. Check attached report details."
    time.sleep(40)  # Wait 40 seconds between tests to avoid hitting rate limits
    