import boto3
import os
from botocore.exceptions import ClientError, NoCredentialsError
import json
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSConfig:
    """
    AWS service configurations for medical chatbot
    """

    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize AWS configuration

        Args:
            region (str): AWS region
        """
        self.region = region
        self.session = None
        self.clients = {}
        self._initialize_session()

    def _initialize_session(self):
        """Initialize AWS session"""
        try:
            self.session = boto3.Session(region_name=self.region)
            logger.info(f"AWS session initialized for region: {self.region}")
        except NoCredentialsError as e:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS session: {str(e)}")
            raise

    def get_client(self, service_name: str):
        """
        Get AWS service client

        Args:
            service_name (str): Name of the AWS service

        Returns:
            AWS service client
        """
        if service_name not in self.clients:
            try:
                self.clients[service_name] = self.session.client(service_name)
                logger.info(f"Created {service_name} client")
            except Exception as e:
                logger.error(f"Failed to create {service_name} client: {str(e)}")
                raise

        return self.clients[service_name]

    def get_bedrock_client(self):
        """Get Amazon Bedrock client"""
        return self.get_client('bedrock-runtime')

    def get_textract_client(self):
        """Get Amazon Textract client"""
        return self.get_client('textract')

    def get_comprehend_client(self):
        """Get Amazon Comprehend Medical client"""
        return self.get_client('comprehendmedical')

    def get_s3_client(self):
        """Get Amazon S3 client"""
        return self.get_client('s3')

    def get_lambda_client(self):
        """Get AWS Lambda client"""
        return self.get_client('lambda')

    def get_lex_client(self):
        """Get Amazon Lex client"""
        return self.get_client('lexv2-models')

    def get_iam_client(self):
        """Get AWS IAM client"""
        return self.get_client('iam')

    def get_apigateway_client(self):
        """Get API Gateway client"""
        return self.get_client('apigateway')

    def create_s3_bucket(self, bucket_name: str) -> bool:
        """
        Create S3 bucket

        Args:
            bucket_name (str): Name of the S3 bucket

        Returns:
            bool: True if successful
        """
        try:
            s3_client = self.get_s3_client()

            if self.region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )

            # Enable versioning
            s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            # Set up CORS for web access
            cors_config = {
                'CORSRules': [
                    {
                        'AllowedHeaders': ['*'],
                        'AllowedMethods': ['GET', 'PUT', 'POST'],
                        'AllowedOrigins': ['*'],
                        'MaxAgeSeconds': 3000
                    }
                ]
            }

            s3_client.put_bucket_cors(
                Bucket=bucket_name,
                CORSConfiguration=cors_config
            )

            logger.info(f"S3 bucket {bucket_name} created successfully")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyExists':
                logger.info(f"S3 bucket {bucket_name} already exists")
                return True
            else:
                logger.error(f"Failed to create S3 bucket: {str(e)}")
                return False

    def upload_to_s3(self, bucket_name: str, file_path: str, s3_key: str) -> bool:
        """
        Upload file to S3 bucket

        Args:
            bucket_name (str): Name of the S3 bucket
            file_path (str): Local file path
            s3_key (str): S3 object key

        Returns:
            bool: True if successful
        """
        try:
            s3_client = self.get_s3_client()
            s3_client.upload_file(file_path, bucket_name, s3_key)
            logger.info(f"File uploaded to S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {str(e)}")
            return False

    def extract_text_from_document(self, bucket_name: str, document_name: str) -> Dict:
        """
        Extract text from document using Textract

        Args:
            bucket_name (str): S3 bucket name
            document_name (str): Document name in S3

        Returns:
            Dict: Extracted text data
        """
        try:
            textract_client = self.get_textract_client()

            response = textract_client.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': document_name
                    }
                }
            )

            logger.info(f"Text extracted from document: {document_name}")
            return response

        except Exception as e:
            logger.error(f"Failed to extract text from document: {str(e)}")
            return {}

    def analyze_medical_text(self, text: str) -> Dict:
        """
        Analyze medical text using Comprehend Medical

        Args:
            text (str): Medical text to analyze

        Returns:
            Dict: Analysis results
        """
        try:
            comprehend_client = self.get_comprehend_client()

            # Detect entities
            entities_response = comprehend_client.detect_entities_v2(Text=text)

            # Detect PHI (Personal Health Information)
            phi_response = comprehend_client.detect_phi(Text=text)

            result = {
                'entities': entities_response,
                'phi': phi_response
            }

            logger.info("Medical text analysis completed")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze medical text: {str(e)}")
            return {}

# AWS Configuration Settings
AWS_CONFIG = {
    "region": "us-east-1",
    "s3_bucket_prefix": "medical-chatbot-documents",
    "lambda_runtime": "python3.9",
    "lambda_timeout": 300,
    "lambda_memory": 1024,
    "bedrock_model_id": "anthropic.claude-haiku-4-5-20251001-v1:0"
}

# Environment variables
def get_env_variables() -> Dict[str, str]:
    """Get environment variables for AWS services"""
    return {
        "AWS_REGION": AWS_CONFIG["region"],
        "S3_BUCKET": os.getenv('S3_BUCKET', ''),
        "PINECONE_API_KEY": os.getenv('PINECONE_API_KEY', ''),
        "PINECONE_INDEX_NAME": os.getenv('PINECONE_INDEX_NAME', 'medical-chatbot-index'),
        "PINECONE_ENVIRONMENT": os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws'),
        "SLACK_BOT_TOKEN": os.getenv('SLACK_BOT_TOKEN', ''),
        "SLACK_SIGNING_SECRET": os.getenv('SLACK_SIGNING_SECRET', ''),
        "BEDROCK_MODEL_ID": AWS_CONFIG["bedrock_model_id"]
    }

def initialize_aws_services(region: str = "us-east-1") -> AWSConfig:
    """
    Initialize AWS services for medical chatbot

    Args:
        region (str): AWS region

    Returns:
        AWSConfig: Configured AWS service manager
    """
    try:
        aws_config = AWSConfig(region=region)
        logger.info("AWS services initialized successfully")
        return aws_config
    except Exception as e:
        logger.error(f"Failed to initialize AWS services: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize AWS services
    aws_services = initialize_aws_services()

    # Test connections
    try:
        bedrock_client = aws_services.get_bedrock_client()
        s3_client = aws_services.get_s3_client()
        print("AWS services initialized and tested successfully!")
    except Exception as e:
        print(f"Error testing AWS services: {str(e)}")