import mlflow
import dagshub



mlflow.set_tracking_uri('https://dagshub.com/sandeeprairai/mlops_mini_project2.mlflow')
dagshub.init(repo_owner='sandeeprairai', repo_name='mlops_mini_project2', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)