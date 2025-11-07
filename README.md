# Predicting Sleep Efficiency with Machine Learning

A project to build and deploy a machine learning model that predicts an individual's sleep efficiency based on their sleep patterns and lifestyle habits.

## Overview

This project explores the relationship between lifestyle habits and sleep quality. It uses the [Sleep Efficiency Dataset from Kaggle](https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency) to build and deploy a machine learning model capable of predicting an individual's sleep efficiency.

The final model, an **XGBoost Regressor**, is served via a FastAPI application and is also deployable as an AWS Lambda function.


## Problem Statement

Sleep is a critical pillar of physical health, cognitive function, and emotional well-being. In our fast-paced society, many individuals struggle not only with the *quantity* of sleep but, more importantly, with its **quality**. Simply being in bed for eight hours does not guarantee restorative rest.

The relationship between daily habits and sleep quality is complex and deeply personal. Factors like caffeine intake, alcohol consumption, exercise frequency, and smoking all interact in non-obvious ways to affect how well we sleep.

A more precise metric for sleep quality is **Sleep Efficiency**—the percentage of time spent in bed that one is actually asleep. A high efficiency score is a key indicator of good sleep health. However, most people lack a clear understanding of which specific lifestyle choices are impacting their sleep efficiency.

## The Objective

This project aims to solve this problem by building a machine learning model. The model will **predict an individual's sleep efficiency score** (a value between 0 and 1) based on a set of their lifestyle and sleep-related habits.

The ultimate goal is to create a predictive tool that could power a wellness application, providing users with actionable, data-driven insights to help them understand and improve their sleep habits.


In this project, our aim is to predict sleep efficiency of individuals based on their sleep habits and lifestyle habits. 



## The Dataset

The original dataset is from [Kaggle: Sleep Efficiency Dataset](https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency), with an original shape of (452, 14).

### Feature Descriptions

| Feature | Description |
| :--- | :--- |
| **Age** | Age of the subject. |
| **Gender** | "Male" or "Female". |
| **Bedtime** | The time the subject went to bed. |
| **Wakeup time** | The time the subject woke up. |
| **Sleep duration** | Total hours slept. |
| **Sleep efficiency** | **(Target)** Proportion of time in bed spent asleep (0-1). |
| **REM sleep percentage** | Percentage of time in REM sleep. |
| **Deep sleep percentage**| Percentage of time in deep sleep. |
| **Light sleep percentage**| Percentage of time in light sleep. |
| **Awakenings** | Number of times the subject woke up during the night. |
| **Caffeine consumption**| Caffeine (mg) consumed in the 24 hours prior. |
| **Alcohol consumption** | Alcohol (ounces) consumed in the 24 hours prior. |
| **Smoking status** | "Yes" or "No". |
| **Exercise frequency** | Times per week the subject exercises. |

-----




## Project Workflow


### 1. Exploratory Data Analysis (EDA)

In the [Jupyter Notebook (`notebook/notebook.ipynb`)](https://www.google.com/search?q=%5Bhttps://github.com/hyunwoooh5/sleep-efficiency/blob/main/notebook/notebook.ipynb%5D\(https://github.com/hyunwoooh5/sleep-efficiency/blob/main/notebook/notebook.ipynb\)), summary statistics were examined, missing values were imputed using mean and mode values, and variable correlations were analyzed.

Most variables were found to be largely independent, except for a notable correlation between `light_sleep_percentage` and `deep_sleep_percentage`, and `light_sleep_percentage` and `sleep_efficiency`.

### 2. Model Selection

Three different types of models were trained and evaluated to predict `sleep_efficiency`. The performance of all models was measured using the **Mean Squared Error (MSE)**.

  * **Ridge Regression ($L_2$):** A linear model with $L_2$ regularization. The best MSE achieved was `0.00407` (with `alpha=0.5`).
  * **Tree-Based Models:** Decision Tree, Random Forest, and XGBoost were tested. The best-performing model was **XGBoost**, which yielded an MSE of **`0.00262`**.
  * **Neural Network:** A fully connected network with 4 hidden layers, 32 neurons each, CELU activation, and a 0.25 dropout rate. This model achieved an MSE of `0.00267`.

### 3. Final Model

The final selected model is an **XGBoost Regressor** with the following hyperparameters, as it provided the lowest MSE:

  * `n_estimators=50`
  * `max_depth=2`
  * `learning_rate=5e-2`

The trained model is saved as `bin/model.pkl`.

-----




## Usage and Deployment

This project uses [**uv**](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

First, install `uv` (on macOS/Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


### Option 1: Run Locally (without Docker)

1.  **Install dependencies** from the lockfile:

    ```bash
    uv sync --locked
    ```

2.  **Run the FastAPI server:**

    ```bash
    uv run python src/serve.py
    ```

3.  The service will be available with a Swagger UI at [http://localhost:9696/docs](https://www.google.com/search?q=http://localhost:9696/docs).


### Option 2: Run Locally (with Docker)

1.  **Build the Docker image:**

    ```bash
    docker build  --platform=linux/amd64 -t sleep-efficiency .
    ```

2.  **Run the container,** mapping port 9696:

    ```bash
    docker run -it --rm --platform=linux/amd64 -p 9696:9696 sleep-efficiency
    ```

3.  Access the service at [http://localhost:9696/docs](https://www.google.com/search?q=http://localhost:9696/docs).


### Retrain the Model

If you make changes to the model or parameters in `src/train.py`, you can retrain the model by running:

```bash
uv run python src/train.py
```

This will overwrite the `bin/model.pkl` file with the newly trained model.




-----


## Deployment to AWS Lambda

This guide assumes you have the AWS CLI configured with the necessary permissions.

#### 1. Prerequisite: IAM Role Setup

The Lambda function needs an execution role.

First, create a trust policy JSON file named `trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Now, create the role and attach the basic Lambda execution policy:

```bash
# Create the role
aws iam create-role \
  --role-name lambda-basic-execution-role \
  --assume-role-policy-document file://trust-policy.json

# Attach the policy
aws iam attach-role-policy \
  --role-name lambda-basic-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```


#### 2. Deploy the Function

The [deploy_lambda.sh](https://github.com/hyunwoooh5/sleep-efficiency/blob/main/deploy_lambda.sh) script handles building the `Dockerfile.lambda` image, pushing it to ECR, and creating/updating the Lambda function.

```bash
./deploy_lambda.sh
```

#### 3. Invoke the Deployed Function

You can test the live Lambda function using `curl` (or a tool like `src/invoke.py`).

**Example `curl` request:**

```bash
curl -X POST 'https://4rrzhd6ucsbgzxieipazjcst4y0cibng.lambda-url.us-east-1.on.aws/' \
-H "Content-Type: application/json" \
-d '{"customer":{"age":65,"gender":"Female","sleep_duration":6.0,"deep_sleep_percentage":70,"light_sleep_percentage":12,"awakenings":0,"caffeine_consumption":0,"alcohol_consumption":0,"smoking_status":"Yes","exercise_frequency":3.0,"sleep_time":1.0,"getup_time":7}}'
```

-----




## Project structure
```
├── bin
│   └── model.pkl                     # Trained final model
├── data
│   └── sleep-efficiency_cleaned.csv  # Cleaned dataset
├── deploy_lambda.sh                  # Deployment script for AWS lambda
├── Dockerfile                        # Docker configuration for FastAPI app
├── Dockerfile.lambda                 # Docker configuration for AWS lambda
├── notebook
│   ├── notebook.ipynb                # Main notebook for EDA and model selection
│   ├── pyproject.toml                # Dependencies for the notebook
│   ├── uv.lock                       # Lockfile for notebook reproducibility
│   └── .python-version               # Python version for notebook reproducibility
├── pyproject.toml                    # Project dependencies (for src)
├── README.md                         # Project documentation
├── src
│   ├── invoke.py                     # Script to invoke AWS lambda
│   ├── lambda_function.py            # AWS lambda handler
│   ├── predict.py                    # Script to load model and make predictions
│   ├── serve.py                      # FastAPI server to serve the model
│   └── train.py                      # Script to train the final model
├── uv.lock                           # Lockfile for project reproducibility
├── .gitignore                        # Ignored files and directories
└── .python-version                   # Python version for deployment reproducibility

```