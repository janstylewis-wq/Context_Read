"""S3-compatible object storage (MinIO in dev, S3/R2 in prod). The browser uploads and
downloads directly through presigned URLs, as in the recorded upload flow."""

from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .config import get_settings


@lru_cache
def get_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key_id,
        aws_secret_access_key=s.s3_secret_access_key,
        region_name=s.s3_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def bucket() -> str:
    return get_settings().s3_bucket


def ensure_bucket() -> None:
    client = get_client()
    try:
        client.head_bucket(Bucket=bucket())
    except ClientError:
        client.create_bucket(Bucket=bucket())


def presign_put(key: str, content_type: str, expires: int = 600) -> str:
    return get_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket(), "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
    )


def presign_get(key: str, expires: int = 86400) -> str:
    return get_client().generate_presigned_url(
        "get_object", Params={"Bucket": bucket(), "Key": key}, ExpiresIn=expires
    )


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    get_client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)


def delete_key(key: str) -> None:
    get_client().delete_object(Bucket=bucket(), Key=key)


def delete_prefix(prefix: str) -> int:
    client, deleted = get_client(), 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket(), Delete={"Objects": keys, "Quiet": True})
            deleted += len(keys)
    return deleted


def get_bytes(key: str) -> bytes:
    return get_client().get_object(Bucket=bucket(), Key=key)["Body"].read()
