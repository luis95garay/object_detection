import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts

expr_name = "object_detection"  

s3_bucket = f"s3://elvis-s3-mlflow/mlruns/{expr_name}"  

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.get_experiment_by_name(expr_name)
if not experiment:
    mlflow.create_experiment(expr_name, s3_bucket)

mlflow.set_experiment(expr_name)

# yolov4tiny fp32
mlflow.start_run('e382fb50896542dc8bb8c25c8538999f')
# log_metric('fps', 2.9)
log_metric('fps-i15', 20.5)
mlflow.end_run()

# yolov4tiny fp16
mlflow.start_run('ed07dd096d0f461ebe88b69bd1921b9a')
# log_metric('fps', 4.5)
log_metric('fps-i15', 26)
mlflow.end_run()

# yolov4 fp32
mlflow.start_run('e0506657835d451a8b99c954b2be6090')
# log_metric('fps', 0.5)
log_metric('fps-i15', 6.5)
mlflow.end_run()

# yolov4 fp16
mlflow.start_run('32381c029e1b4d41bc7209f8b13e89a6')
# log_metric('fps', 1.0)
log_metric('fps-i15', 10.5)
mlflow.end_run()


# resnet10_300px_v1_fp16
mlflow.start_run('3df90b5e2c7d48d7bac5803524eced17')
# log_metric('fps', 22.78)
log_metric('fps-i15', 31)
mlflow.end_run()

# resnet10_300px_v1_fp32
mlflow.start_run('64b6f7b983864931a88b766a90072562')
# log_metric('fps', 12.79)
log_metric('fps-i15', 31)
mlflow.end_run()

# yolov7_1280px_v2_fp16
mlflow.start_run('4bc95ef9a4bd45ecb6a1b434316c9b92')
# log_metric('fps', 0.4)
log_metric('fps-i15', 5.8)
mlflow.end_run()

# yolov7_1280px_v2_fp32
mlflow.start_run('2c1a00c0936f4bc0bfdac0b2d4719471')
# log_metric('fps', 0.25)
log_metric('fps-i15', 3.2)
mlflow.end_run()

# yolov7tiny_1280px_v2_fp16
mlflow.start_run('5606d5a8a71b490abc0e9c5baa112871')
# log_metric('fps', 2.65)
log_metric('fps-i15', 21)
mlflow.end_run()

# yolov7tiny_1280px_v2_fp32
mlflow.start_run('39d3c926757c423d8fc1eb55405bd021')
# log_metric('fps', 1.6)
log_metric('fps-i15', 15.5)
mlflow.end_run()