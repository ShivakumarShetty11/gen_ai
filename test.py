import mlflow
import google.generativeai as genai
import os
import pandas as pd
import dagshub

# Configure Gemini API key directly in code
GEMINI_API_KEY = "AIzaSyCpHNl_rmvp-AMRn1KNPPxC4G1UMCmfGHc"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

dagshub.init(repo_owner='@shivakumar.vahani', repo_name='MLfLow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/shivakumar.vahani/MLflow")

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)

# Custom Gemini model wrapper for MLflow
class GeminiModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name="gemini-1.5-flash", system_prompt=""):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model = genai.GenerativeModel(model_name)
    
    def predict(self, context, model_input):
        responses = []
        for question in model_input["inputs"]:
            prompt = f"{self.system_prompt}\n\nQuestion: {question}"
            response = self.model.generate_content(prompt)
            responses.append(response.text)
        return responses

mlflow.set_experiment("LLM Evaluation")
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    
    # Create and log Gemini model
    gemini_model = GeminiModel(model_name="gemini-1.5-flash", system_prompt=system_prompt)
    
    # Log the model
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=gemini_model,
        pip_requirements=["google-generativeai", "pandas", "mlflow", "dagshub"]
    )

    # Use predefined question-answering metrics to evaluate our model.
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(), mlflow.metrics.genai.answer_similarity()]
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    df = pd.DataFrame(eval_table)
    df.to_csv('eval.csv')
    print(f"See evaluation table below: \n{eval_table}")