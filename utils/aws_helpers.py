import boto3
import json
import logging
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError, NoCredentialsError
import os
import time

# Configure logging
logger = logging.getLogger(__name__)

class AWSHelpers:
    """Helper class for AWS operations"""

    def __init__(self, region: str = None):
        """
        Initialize AWS helpers

        Args:
            region (str): AWS region
        """
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self.clients = {}

    def get_aws_client(self, service_name: str, region: str = None):
        """
        Get AWS service client with caching

        Args:
            service_name (str): AWS service name
            region (str): AWS region (optional)

        Returns:
            AWS service client
        """
        region = region or self.region
        client_key = f"{service_name}_{region}"

        if client_key not in self.clients:
            try:
                self.clients[client_key] = boto3.client(
                    service_name, 
                    region_name=region
                )
                logger.info(f"Created {service_name} client for region {region}")
            except NoCredentialsError:
                logger.error("AWS credentials not found")
                raise
            except Exception as e:
                logger.error(f"Failed to create {service_name} client: {str(e)}")
                raise

        return self.clients[client_key]

def get_aws_client(service_name: str, region: str = None):
    """
    Get AWS service client (convenience function)

    Args:
        service_name (str): AWS service name
        region (str): AWS region

    Returns:
        AWS service client
    """
    helpers = AWSHelpers(region)
    return helpers.get_aws_client(service_name, region)

def upload_to_s3(bucket_name: str, file_path: str, s3_key: str, region: str = None) -> bool:
    """
    Upload file to S3 bucket

    Args:
        bucket_name (str): S3 bucket name
        file_path (str): Local file path
        s3_key (str): S3 object key
        region (str): AWS region

    Returns:
        bool: True if successful
    """
    try:
        s3_client = get_aws_client('s3', region)
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logger.info(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        return True
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {str(e)}")
        return False

def download_from_s3(bucket_name: str, s3_key: str, local_path: str, region: str = None) -> bool:
    """
    Download file from S3 bucket

    Args:
        bucket_name (str): S3 bucket name
        s3_key (str): S3 object key
        local_path (str): Local file path to save
        region (str): AWS region

    Returns:
        bool: True if successful
    """
    try:
        s3_client = get_aws_client('s3', region)
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"Successfully downloaded s3://{bucket_name}/{s3_key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download file from S3: {str(e)}")
        return False

def list_s3_objects(bucket_name: str, prefix: str = '', region: str = None) -> List[str]:
    """
    List objects in S3 bucket

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): Object prefix filter
        region (str): AWS region

    Returns:
        List[str]: List of S3 object keys
    """
    try:
        s3_client = get_aws_client('s3', region)
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []

    except Exception as e:
        logger.error(f"Failed to list S3 objects: {str(e)}")
        return []

def extract_text_from_document(bucket_name: str, document_name: str, region: str = None) -> Dict[str, Any]:
    """
    Extract text from document using Amazon Textract

    Args:
        bucket_name (str): S3 bucket name
        document_name (str): Document name in S3
        region (str): AWS region

    Returns:
        Dict: Extracted text data with blocks and confidence scores
    """
    try:
        textract_client = get_aws_client('textract', region)

        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': document_name
                }
            }
        )

        # Process blocks to extract text
        extracted_text = []
        confidence_scores = []

        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                extracted_text.append(block.get('Text', ''))
                confidence_scores.append(block.get('Confidence', 0.0))

        result = {
            'text': '\n'.join(extracted_text),
            'blocks': response.get('Blocks', []),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'total_blocks': len(response.get('Blocks', [])),
            'line_count': len(extracted_text)
        }

        logger.info(f"Successfully extracted text from {document_name} with {result['line_count']} lines")
        return result

    except Exception as e:
        logger.error(f"Failed to extract text from document: {str(e)}")
        return {
            'text': '',
            'blocks': [],
            'average_confidence': 0.0,
            'total_blocks': 0,
            'line_count': 0,
            'error': str(e)
        }

def analyze_medical_text(text: str, region: str = None) -> Dict[str, Any]:
    """
    Analyze medical text using Amazon Comprehend Medical

    Args:
        text (str): Medical text to analyze
        region (str): AWS region

    Returns:
        Dict: Analysis results including entities, PHI, and relationships
    """
    try:
        comprehend_client = get_aws_client('comprehendmedical', region)

        # Detect medical entities
        entities_response = comprehend_client.detect_entities_v2(Text=text)

        # Detect PHI (Personal Health Information)
        try:
            phi_response = comprehend_client.detect_phi(Text=text)
        except Exception as e:
            logger.warning(f"PHI detection failed: {str(e)}")
            phi_response = {'Entities': []}

        # Extract relevant information
        medications = []
        conditions = []
        symptoms = []
        dosages = []

        for entity in entities_response.get('Entities', []):
            category = entity.get('Category', '')
            text_content = entity.get('Text', '')
            confidence = entity.get('Score', 0.0)

            if category == 'MEDICATION':
                medications.append({
                    'text': text_content,
                    'confidence': confidence,
                    'attributes': entity.get('Attributes', [])
                })
            elif category == 'MEDICAL_CONDITION':
                conditions.append({
                    'text': text_content,
                    'confidence': confidence,
                    'attributes': entity.get('Attributes', [])
                })
            elif category == 'SYMPTOM':
                symptoms.append({
                    'text': text_content,
                    'confidence': confidence
                })
            elif category == 'DOSAGE':
                dosages.append({
                    'text': text_content,
                    'confidence': confidence
                })

        result = {
            'entities': entities_response,
            'phi': phi_response,
            'summary': {
                'medications': medications,
                'conditions': conditions,
                'symptoms': symptoms,
                'dosages': dosages,
                'medication_count': len(medications),
                'condition_count': len(conditions),
                'symptom_count': len(symptoms),
                'dosage_count': len(dosages)
            }
        }

        logger.info(f"Medical text analysis completed: {len(medications)} medications, {len(conditions)} conditions found")
        return result

    except Exception as e:
        logger.error(f"Failed to analyze medical text: {str(e)}")
        return {
            'entities': {'Entities': []},
            'phi': {'Entities': []},
            'summary': {
                'medications': [],
                'conditions': [],
                'symptoms': [],
                'dosages': [],
                'medication_count': 0,
                'condition_count': 0,
                'symptom_count': 0,
                'dosage_count': 0
            },
            'error': str(e)
        }

def invoke_bedrock_model(model_id: str, prompt: str, region: str = None) -> Dict[str, Any]:
    """
    Invoke Amazon Bedrock model

    Args:
        model_id (str): Bedrock model ID
        prompt (str): Input prompt
        region (str): AWS region

    Returns:
        Dict: Model response
    """
    try:
        bedrock_client = get_aws_client('bedrock-runtime', region)

        # Prepare request body based on model
        if 'anthropic' in model_id.lower():
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1,
                "anthropic_version": "bedrock-2023-05-31"
            }
        else:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 2000,
                    "temperature": 0.1,
                    "topP": 0.9
                }
            }

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )

        # Parse response
        response_body = json.loads(response['body'].read())

        result = {
            'response': response_body,
            'model_id': model_id,
            'success': True
        }

        logger.info(f"Successfully invoked Bedrock model: {model_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to invoke Bedrock model: {str(e)}")
        return {
            'response': {},
            'model_id': model_id,
            'success': False,
            'error': str(e)
        }

def send_lambda_invocation(function_name: str, payload: Dict, region: str = None) -> Dict[str, Any]:
    """
    Invoke Lambda function

    Args:
        function_name (str): Lambda function name
        payload (Dict): Function payload
        region (str): AWS region

    Returns:
        Dict: Function response
    """
    try:
        lambda_client = get_aws_client('lambda', region)

        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        # Parse response
        response_payload = json.loads(response['Payload'].read())

        result = {
            'status_code': response['StatusCode'],
            'payload': response_payload,
            'success': response['StatusCode'] == 200
        }

        logger.info(f"Successfully invoked Lambda function: {function_name}")
        return result

    except Exception as e:
        logger.error(f"Failed to invoke Lambda function: {str(e)}")
        return {
            'status_code': 500,
            'payload': {},
            'success': False,
            'error': str(e)
        }

def create_presigned_url(bucket_name: str, object_key: str, expiration: int = 3600, region: str = None) -> Optional[str]:
    """
    Create presigned URL for S3 object

    Args:
        bucket_name (str): S3 bucket name
        object_key (str): S3 object key
        expiration (int): URL expiration time in seconds
        region (str): AWS region

    Returns:
        Optional[str]: Presigned URL or None if failed
    """
    try:
        s3_client = get_aws_client('s3', region)

        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )

        logger.info(f"Generated presigned URL for s3://{bucket_name}/{object_key}")
        return url

    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {str(e)}")
        return None

def get_secret_value(secret_name: str, region: str = None) -> Optional[str]:
    """
    Get secret value from AWS Secrets Manager

    Args:
        secret_name (str): Secret name
        region (str): AWS region

    Returns:
        Optional[str]: Secret value or None if failed
    """
    try:
        secrets_client = get_aws_client('secretsmanager', region)

        response = secrets_client.get_secret_value(SecretId=secret_name)

        logger.info(f"Successfully retrieved secret: {secret_name}")
        return response['SecretString']

    except Exception as e:
        logger.error(f"Failed to retrieve secret: {str(e)}")
        return None

# Convenience functions for common operations
def quick_textract(bucket: str, key: str) -> str:
    """Quick text extraction from S3 document"""
    result = extract_text_from_document(bucket, key)
    return result.get('text', '')

def quick_medical_analysis(text: str) -> Dict:
    """Quick medical text analysis"""
    result = analyze_medical_text(text)
    return result.get('summary', {})