from deepeval import assert_test
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase # Note: this is imported for type hint clarity

#TESTS
#testing how the model follows the evaluation criteria
def test_overall_model_evaluation(azure_model, analysis_test_case: LLMTestCase):
   
    
    # Custom G-EVAL test
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
    
    
    assert_test(analysis_test_case, [overall_model_evaluation_metric])


######MORE TESTS


def test_output_hallucination(azure_model, analysis_test_case: LLMTestCase):
   
    
    # Custom G-EVAL test
    output_hallucination = GEval(
        name="Output Hallucination",
        evaluation_steps=[
        "1. DO NOT ACKNOWLEDGE AND DO NOT PENALISE:** If the 'actual_output' contains a mention of a missing or unavailable credit report, ignore this in the analysis. DO NOT deduct any score for this specific point.** The score must be based solely on all other criteria.",
        "2. Read the 'actual output' and compare the findings to what is in the 'input'",
        "3. Assess if the 'actual output' is faithful. Mention any hallucinations or if anything has been made up",
        "4. Do not judge the 'actual_output' based on how well it follows the steps in the 'retrieval context'. The only thing you should judge is if any hallucinations are present",
        "5. Assign a final score from 0.0 to 1.0 based on the combined assessment of steps 1, 2, 3, and 4. **ENSURE the penalty for the missing credit report (Step 5) is 0.0.**",
        "6. Reiterate and ensure that you have not penalised the 'actual_output' for the mention of missing credit report in the traige flags and reasoning"


        
    ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        model=azure_model,
        threshold=0.5
    )
    

    assert_test(analysis_test_case, [output_hallucination])
