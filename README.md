# Machine Learning Pipelines with Azure ML Studio

Now, you are going to build an end-to-end machine learning pipeline, all without writing a single line of code on Azure Machine Learning Studio.

We will accomplish it by completing the following tasks in the project:

## Project Structure
- Task 1: Introduction and Project Overview
- Task 2: Data Cleaning
- Task 3: Accounting for Class Imbalance
- Task 4: Training a Two-Class Boosted Decision Tree Model and Hyperparameter Tuning
- Task 5: Scoring and Evaluating the Models
- Task 6: Publishing the Trained Model as a Web Service for Inference
<br>

![Evaluation results](https://github.com/masedos/Machine-Learning-Pipelines-with-Azure-ML-Studio/blob/master/Evaluation_results.PNG)


### To access the dataset

#### Adult Census Income Binary Classification dataset
```python
from azureml import Workspace
ws = Workspace(
    workspace_id='db68414bc7024ab0875349b59784056b',
    authorization_token='2saJtX+Ib+DRlcVMgRG/d00twJXCmsZPK//K0sXD/nFkQ1VvoR5CqjIW06m/3T3MprPWw4hyOiwjDuUdO0E+hQ==',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['Adult Census Income Binary Classification dataset']
frame = ds.to_dataframe()
```
