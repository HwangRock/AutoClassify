from django.apps import AppConfig
from .utils import download_model_from_s3

class YourAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'yourapp'

    def ready(self):
        bucket_name = '나중에 지정할 버킷이름'
        s3_key = 's3키 이름'
        local_path = '로컬에 저장할 모델 경로'

        download_model_from_s3(bucket_name, s3_key, local_path)
