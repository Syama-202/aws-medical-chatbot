import boto3
import json
import zipfile
import os
import time
from botocore.exceptions import ClientError
import logging
import sys
from pathlib import Path

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent))

from config.aws_config import AWSConfig, initialize_aws_services
from config.pinecone_setup import initialize_pinecone
from config.slack_config import initialize_slack_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalChatbotDeployer:
    def __init__(self, region='us-east-1'):
        """
        Initialize the Medical Chatbot Deployer

        Args:
            region (str): AWS region for deployment
        """
        self.region = region
        self.aws_config = initialize_aws_services(region)

        # Get AWS clients
        self.lambda_client = self.aws_config.get_lambda_client()
        self.iam_client = self.aws_config.get_iam_client()
        self.s3_client = self.aws_config.get_s3_client()
        self.lex_client = self.aws_config.get_lex_client()
        self.apigateway_client = self.aws_config.get_apigateway_client()

    def create_lambda_execution_role(self, role_name):
        """Create IAM role for Lambda functions"""
        trust_policy = {
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

        try:
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Path='/service-role/'
            )

            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/AmazonTextractFullAccess',
                'arn:aws:iam::aws:policy/ComprehendMedicalFullAccess',
                'arn:aws:iam::aws:policy/AmazonBedrockFullAccess',
                'arn:aws:iam::aws:policy/AmazonLexFullAccess'
            ]

            for policy_arn in policies:
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )

            # Wait for role to be available
            time.sleep(10)
            logger.info(f"✅ Created IAM role: {role_name}")
            return response['Role']['Arn']

        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                response = self.iam_client.get_role(RoleName=role_name)
                logger.info(f"🔄 Using existing IAM role: {role_name}")
                return response['Role']['Arn']
            else:
                logger.error(f"❌ Failed to create IAM role: {str(e)}")
                raise

    def create_lambda_deployment_package(self, function_code, function_name, requirements=None):
        """Create deployment package for Lambda function"""
        zip_filename = f"{function_name}.zip"

        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            # Add the main function code
            zip_file.writestr('lambda_function.py', function_code)

            # Add requirements if they exist
            if requirements:
                zip_file.writestr('requirements.txt', requirements)

            # Add config files
            config_files = [
                'config/aws_config.py',
                'config/pinecone_setup.py', 
                'config/slack_config.py',
                'utils/aws_helpers.py',
                'utils/logging_config.py'
            ]

            for config_file in config_files:
                if os.path.exists(config_file):
                    zip_file.write(config_file, os.path.basename(config_file))

        with open(zip_filename, 'rb') as zip_file:
            zip_content = zip_file.read()

        # Clean up
        os.remove(zip_filename)
        return zip_content

    def deploy_lambda_function(self, function_name, function_code, role_arn, environment_vars, requirements=None):
        """Deploy a Lambda function"""
        zip_content = self.create_lambda_deployment_package(function_code, function_name, requirements)

        try:
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Environment={'Variables': environment_vars},
                Timeout=300,
                MemorySize=1024,
                Tags={
                    'Project': 'MedicalChatbot',
                    'Environment': 'Production'
                }
            )

            logger.info(f"✅ Lambda function {function_name} created successfully")
            return response['FunctionArn']

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException':
                # Update existing function
                self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )

                self.lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Environment={'Variables': environment_vars},
                    Timeout=300,
                    MemorySize=1024
                )

                logger.info(f"🔄 Lambda function {function_name} updated successfully")

                response = self.lambda_client.get_function(FunctionName=function_name)
                return response['Configuration']['FunctionArn']
            else:
                logger.error(f"❌ Failed to deploy Lambda function {function_name}: {str(e)}")
                raise

    def create_s3_bucket(self, bucket_name):
        """Create S3 bucket for medical documents"""
        return self.aws_config.create_s3_bucket(bucket_name)

    def deploy_lex_bot(self, bot_config_path='config/lex_bot_config.json'):
        """Deploy Lex bot for medical chatbot"""
        try:
            with open(bot_config_path, 'r') as f:
                bot_config = json.load(f)

            # Create bot
            response = self.lex_client.create_bot(**bot_config)
            bot_id = response['botId']

            logger.info(f"✅ Lex bot created successfully: {bot_id}")

            # Wait for bot to be available
            time.sleep(30)

            # Build bot
            build_response = self.lex_client.build_bot_locale(
                botId=bot_id,
                botVersion='DRAFT',
                localeId='en_US'
            )

            logger.info(f"✅ Lex bot build initiated: {build_response['buildId']}")

            return bot_id

        except ClientError as e:
            logger.error(f"❌ Failed to deploy Lex bot: {str(e)}")
            return None

    def setup_api_gateway(self, lambda_function_arn):
        """Set up API Gateway for Slack integration"""
        try:
            # Create REST API
            api_response = self.apigateway_client.create_rest_api(
                name='medical-chatbot-slack-api',
                description='API Gateway for Medical Chatbot Slack integration',
                endpointConfiguration={
                    'types': ['REGIONAL']
                }
            )

            api_id = api_response['id']
            logger.info(f"✅ API Gateway created: {api_id}")

            return api_id

        except ClientError as e:
            logger.error(f"❌ Failed to create API Gateway: {str(e)}")
            return None

    def deploy_all_functions(self, config):
        """Deploy all Lambda functions and services"""
        logger.info("🚀 Starting Medical Chatbot deployment...")

        # Initialize external services
        try:
            pinecone_db = initialize_pinecone()
            slack_client = initialize_slack_client()
            logger.info("✅ External services initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize external services: {str(e)}")
            raise

        # Create IAM role
        role_arn = self.create_lambda_execution_role('medical-chatbot-lambda-role')

        # Create S3 bucket
        bucket_name = self.create_s3_bucket(config['s3_bucket'])
        if bucket_name:
            logger.info(f"✅ S3 bucket ready: {bucket_name}")

        # Deploy Lambda functions
        function_arns = {}

        # Read Lambda function codes
        lambda_functions = self.get_lambda_function_codes()

        for func_name, func_data in lambda_functions.items():
            env_vars = self.get_environment_variables(func_name, config, bucket_name)

            try:
                function_arn = self.deploy_lambda_function(
                    func_data['name'],
                    func_data['code'],
                    role_arn,
                    env_vars,
                    func_data.get('requirements')
                )
                function_arns[func_name] = function_arn

            except Exception as e:
                logger.error(f"❌ Failed to deploy {func_name}: {str(e)}")
                continue

        # Deploy Lex bot
        bot_id = self.deploy_lex_bot()
        if bot_id:
            function_arns['lex_bot_id'] = bot_id

        # Set up API Gateway
        if 'slack_handler' in function_arns:
            api_id = self.setup_api_gateway(function_arns['slack_handler'])
            if api_id:
                function_arns['api_gateway_id'] = api_id

        logger.info("🎉 Deployment completed!")
        return function_arns

    def get_environment_variables(self, func_name, config, bucket_name):
        """Get environment variables for each function"""
        base_vars = {
            'AWS_REGION': self.region,
            'S3_BUCKET': bucket_name,
            'PINECONE_API_KEY': config.get('pinecone_api_key', ''),
            'PINECONE_INDEX_NAME': config.get('pinecone_index_name', 'medical-chatbot-index'),
            'PINECONE_ENVIRONMENT': config.get('pinecone_environment', 'us-east-1-aws')
        }

        func_specific_vars = {
            'slack_handler': {
                'SLACK_SIGNING_SECRET': config.get('slack_signing_secret', ''),
                'SLACK_BOT_TOKEN': config.get('slack_bot_token', ''),
                'LEX_BOT_ID': config.get('lex_bot_id', ''),
                'LEX_BOT_ALIAS_ID': config.get('lex_bot_alias_id', 'TSTALIASID'),
                'LEX_LOCALE_ID': 'en_US',
                'DOCUMENT_PROCESSOR_FUNCTION': 'medical-chatbot-document-processor',
                'RAG_PROCESSOR_FUNCTION': 'medical-chatbot-rag-processor'
            },
            'document_processor': {
                'SLACK_BOT_TOKEN': config.get('slack_bot_token', ''),
                'RAG_PROCESSOR_FUNCTION': 'medical-chatbot-rag-processor'
            },
            'rag_processor': {
                'SLACK_BOT_TOKEN': config.get('slack_bot_token', '')
            },
            'lex_fulfillment': {
                'RAG_PROCESSOR_FUNCTION': 'medical-chatbot-rag-processor'
            }
        }

        env_vars = {**base_vars, **func_specific_vars.get(func_name, {})}
        return env_vars

    def get_lambda_function_codes(self):
        """Get Lambda function codes from files"""
        functions = {}

        lambda_dir = Path('lambda_functions')
        if lambda_dir.exists():
            for py_file in lambda_dir.glob('*.py'):
                func_name = py_file.stem.replace('_', '_')
                with open(py_file, 'r') as f:
                    functions[func_name] = {
                        'name': f'medical-chatbot-{func_name.replace("_", "-")}',
                        'code': f.read(),
                        'requirements': self.get_requirements(func_name)
                    }

        return functions

    def get_requirements(self, func_name):
        """Get requirements for each function"""
        base_requirements = """
boto3>=1.26.0
botocore>=1.29.0
pinecone-client>=2.2.0
slack-sdk>=3.20.0
requests>=2.28.0
"""

        func_specific_requirements = {
            'slack_handler': base_requirements,
            'document_processor': base_requirements,
            'rag_processor': base_requirements + "openai>=1.0.0\n",
            'lex_fulfillment': base_requirements
        }

        return func_specific_requirements.get(func_name, base_requirements)

def main():
    """Main deployment function"""
    # Configuration - Replace with your actual values
    config = {
        'slack_signing_secret': os.getenv('SLACK_SIGNING_SECRET', 'your-slack-signing-secret'),
        'slack_bot_token': os.getenv('SLACK_BOT_TOKEN', 'xoxb-your-slack-bot-token'),
        'pinecone_api_key': os.getenv('PINECONE_API_KEY', 'your-pinecone-api-key'),
        'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'medical-chatbot-index'),
        'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws'),
        's3_bucket': f"medical-chatbot-documents-{int(time.time())}"
    }

    # Validate configuration
    required_vars = ['slack_signing_secret', 'slack_bot_token', 'pinecone_api_key']
    missing_vars = [var for var in required_vars if not config[var] or config[var].startswith('your-')]

    if missing_vars:
        logger.error(f"❌ Missing required configuration: {', '.join(missing_vars)}")
        logger.error("Please set environment variables or update the config dictionary")
        return False

    try:
        deployer = MedicalChatbotDeployer()
        function_arns = deployer.deploy_all_functions(config)

        print("\n🎉 Deployment Complete!")
        print("=" * 50)
        print("Function ARNs:")
        for name, arn in function_arns.items():
            print(f"  {name}: {arn}")

        print("\n📋 Next Steps:")
        print("1. Update your Slack app configuration with the API Gateway URL")
        print("2. Test the bot by mentioning it in a Slack channel")
        print("3. Upload a medical document to test document processing")
        print("4. Monitor CloudWatch logs for any issues")

        return True

    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)