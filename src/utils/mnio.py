from minio import Minio
import os
import json

config = os.environ.get("MC_CONFIG")


def load_config(config_path=config):
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)
        endpoint_url = config_data['aliases']['pbs']['url'].replace("https://", "")
        access_key = config_data['aliases']['pbs']['accessKey']
        secret_key = config_data['aliases']['pbs']['secretKey']
    return endpoint_url, access_key, secret_key


def minio_client():
    endpoint_url, access_key, secret_key = load_config()
    client = Minio(endpoint_url, access_key, secret_key, region='eu-central-1')
    return client


def read_dir(bucket, dir, recursive=True):
    client = minio_client()
    objects = client.list_objects(bucket, dir, recursive=recursive)
    return objects


if __name__ == '__main__':
    data_dir = 'data/' + sorted([f.object_name[5:-1] for f in read_dir('pbs', 'data/', False)])[-1]
    print(data_dir)
