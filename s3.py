import boto3
import joblib

# S3 연결
accessKey = "AKIAW36LSHKJAXBX63FE"
secretKey = "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"
s3 = boto3.client('s3', aws_access_key_id=accessKey, aws_secret_access_key=secretKey)

# S3 모델 다운로드
bucket = 'sursubway-bk'
s3_dir = 'xgboost_model.pkl'
local_dir = './xgboost_model.pkl'
s3.download_file(bucket, s3_dir, local_dir)