import json
import logging
import os
import boto3
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import urllib.request

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
comprehend_client = boto3.client('comprehendmedical')
lambda_client = boto3.client('lambda')

# Environment variables
SLACK_BOT_TOKEN = os.environ['SLACK_BOT_TOKEN']
S3_BUCKET = os.environ['S3_BUCKET']
RAG_PROCESSOR_FUNCTION = os.environ['RAG_PROCESSOR_FUNCTION']

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN)

def download_slack_file(file_id, file_name):
    """Download file from Slack using urllib (no requests library needed)"""
    try:
        # Get file info from Slack
        file_info = slack_client.files_info(file=file_id)
        file_data = file_info['file']
        
        # Get private download URL
        file_url = file_data.get('url_private_download') or file_data.get('url_private')
        
        if not file_url:
            raise ValueError("Could not get file download URL from Slack")
        
        # Download file using urllib (built-in, no extra library needed)
        req = urllib.request.Request(
            file_url,
            headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'}
        )
        
        with urllib.request.urlopen(req) as response:
            file_content = response.read()
        
        # Save to S3
        s3_key = f"uploaded_documents/{file_id}_{file_name}"
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType=file_data.get('mimetype', 'application/octet-stream')
        )
        
        logger.info(f"File downloaded and saved to S3: {s3_key}")
        return s3_key
    
    except Exception as e:
        logger.error(f"Error downloading Slack file: {str(e)}")
        raise

def extract_text_with_textract(s3_key):
    """Extract text from document using Amazon Textract"""
    try:
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': S3_BUCKET,
                    'Name': s3_key
                }
            }
        )
        
        # Extract text from blocks
        extracted_text = []
        confidence_scores = []
        
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                text = block.get('Text', '')
                confidence = block.get('Confidence', 0.0)
                
                if text.strip():
                    extracted_text.append(text)
                    confidence_scores.append(confidence)
        
        full_text = '\n'.join(extracted_text)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        result = {
            'text': full_text,
            'line_count': len(extracted_text),
            'average_confidence': avg_confidence,
            'blocks': response.get('Blocks', [])
        }
        
        logger.info(f"Text extraction completed: {len(extracted_text)} lines, {avg_confidence:.2f}% confidence")
        return result
    
    except Exception as e:
        logger.error(f"Error extracting text with Textract: {str(e)}")
        raise

def analyze_medical_text(text):
    """Analyze extracted text with Amazon Comprehend Medical"""
    try:
        # Comprehend Medical has a 20,000 character limit
        # If text is longer, truncate it
        max_length = 20000
        if len(text) > max_length:
            logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Detect medical entities
        entities_response = comprehend_client.detect_entities_v2(Text=text)
        
        # Detect PHI (Personal Health Information)
        try:
            phi_response = comprehend_client.detect_phi(Text=text)
        except Exception as e:
            logger.warning(f"PHI detection failed: {str(e)}")
            phi_response = {'Entities': []}
        
        # Extract and categorize entities
        medications = []
        conditions = []
        dosages = []
        procedures = []
        anatomies = []
        
        for entity in entities_response.get('Entities', []):
            category = entity.get('Category', '')
            text_content = entity.get('Text', '')
            confidence = entity.get('Score', 0.0)
            
            entity_info = {
                'text': text_content,
                'confidence': confidence,
                'type': entity.get('Type', ''),
                'attributes': entity.get('Attributes', [])
            }
            
            if category == 'MEDICATION':
                medications.append(entity_info)
            elif category == 'MEDICAL_CONDITION':
                conditions.append(entity_info)
            elif category == 'TEST_TREATMENT_PROCEDURE':
                procedures.append(entity_info)
            elif category == 'ANATOMY':
                anatomies.append(entity_info)
        
        analysis_result = {
            'entities_response': entities_response,
            'phi_response': phi_response,
            'summary': {
                'medications': medications,
                'conditions': conditions,
                'procedures': procedures,
                'anatomies': anatomies,
                'total_entities': len(entities_response.get('Entities', [])),
                'medication_count': len(medications),
                'condition_count': len(conditions)
            }
        }
        
        logger.info(f"Medical analysis completed: {len(medications)} medications, {len(conditions)} conditions")
        return analysis_result
    
    except Exception as e:
        logger.error(f"Error analyzing medical text: {str(e)}")
        raise

def format_analysis_for_slack(analysis):
    """Format medical analysis results for Slack"""
    try:
        summary = analysis.get('summary', {})
        medications = summary.get('medications', [])
        conditions = summary.get('conditions', [])
        procedures = summary.get('procedures', [])
        
        # Create formatted message
        message_parts = ["📋 *Medical Document Analysis Complete*\n"]
        
        if medications:
            message_parts.append("*💊 Medications Found:*")
            for med in medications[:10]:
                confidence = med.get('confidence', 0) * 100
                message_parts.append(f"• {med['text']} ({confidence:.1f}% confidence)")
            if len(medications) > 10:
                message_parts.append(f"• ... and {len(medications) - 10} more medications")
            message_parts.append("")
        
        if conditions:
            message_parts.append("*🔍 Medical Conditions:*")
            for condition in conditions[:5]:
                confidence = condition.get('confidence', 0) * 100
                message_parts.append(f"• {condition['text']} ({confidence:.1f}% confidence)")
            if len(conditions) > 5:
                message_parts.append(f"• ... and {len(conditions) - 5} more conditions")
            message_parts.append("")
        
        if procedures:
            message_parts.append("*🏥 Procedures:*")
            for procedure in procedures[:5]:
                confidence = procedure.get('confidence', 0) * 100
                message_parts.append(f"• {procedure['text']} ({confidence:.1f}% confidence)")
            if len(procedures) > 5:
                message_parts.append(f"• ... and {len(procedures) - 5} more procedures")
            message_parts.append("")
        
        # Add summary statistics
        total_entities = summary.get('total_entities', 0)
        message_parts.append(f"*📊 Summary:* {total_entities} medical entities detected")
        
        # Add important warnings
        message_parts.extend([
            "",
            "*⚠️ Important Warnings:*",
            "• Always verify medication details with your healthcare provider",
            "• Check for drug interactions and allergies",
            "• This analysis is for informational purposes only",
            "• Consult a medical professional for treatment decisions"
        ])
        
        return "\n".join(message_parts)
    
    except Exception as e:
        logger.error(f"Error formatting analysis for Slack: {str(e)}")
        return f"Analysis completed but formatting failed: {str(e)}"

def send_slack_response(channel, message):
    """Send response back to Slack"""
    try:
        slack_client.chat_postMessage(
            channel=channel,
            text=message,
            mrkdwn=True
        )
        logger.info(f"Response sent to Slack channel: {channel}")
    
    except SlackApiError as e:
        logger.error(f"Error sending Slack message: {e.response['error']}")
    except Exception as e:
        logger.error(f"Unexpected error sending Slack message: {str(e)}")

def invoke_rag_processor(extracted_text, analysis, channel, user):
    """Invoke RAG processor for additional insights"""
    try:
        payload = {
            'extracted_text': extracted_text[:5000],  # Limit payload size
            'medical_analysis': analysis,
            'channel': channel,
            'user': user,
            'action': 'generate_insights'
        }
        
        # Invoke RAG processor asynchronously
        lambda_client.invoke(
            FunctionName=RAG_PROCESSOR_FUNCTION,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        
        logger.info("RAG processor invoked for additional insights")
    
    except Exception as e:
        logger.error(f"Error invoking RAG processor: {str(e)}")
        # Don't fail the main process if RAG processor fails

def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        logger.info(f"Document processor received event: {json.dumps(event)}")
        
        # Extract event data
        file_id = event.get('file_id')
        file_name = event.get('file_name', 'unknown_document')
        channel = event.get('channel')
        user = event.get('user')
        
        if not file_id or not channel:
            raise ValueError("Missing required parameters: file_id or channel")
        
        # Send initial acknowledgment
        send_slack_response(
            channel,
            f"📄 Processing document: *{file_name}*\nPlease wait while I analyze the medical content..."
        )
        
        # Step 1: Download file from Slack to S3
        logger.info(f"Downloading file: {file_name}")
        s3_key = download_slack_file(file_id, file_name)
        
        # Step 2: Extract text using Textract
        logger.info("Extracting text from document")
        text_result = extract_text_with_textract(s3_key)
        
        if not text_result['text'].strip():
            send_slack_response(
                channel,
                "❌ No readable text found in the document. Please ensure the document is clear and contains text."
            )
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'no_text_found',
                    'message': 'No readable text in document'
                })
            }
        
        # Step 3: Analyze medical content
        logger.info("Analyzing medical content")
        medical_analysis = analyze_medical_text(text_result['text'])
        
        # Step 4: Format and send results to Slack
        formatted_response = format_analysis_for_slack(medical_analysis)
        send_slack_response(channel, formatted_response)
        
        # Step 5: Invoke RAG processor for additional insights
        logger.info("Invoking RAG processor for AI insights")
        invoke_rag_processor(
            text_result['text'],
            medical_analysis,
            channel,
            user
        )
        
        logger.info(f"Document processing completed successfully for: {file_name}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'message': 'Document processed successfully',
                'file_name': file_name,
                'entities_found': medical_analysis['summary']['total_entities'],
                'medications': medical_analysis['summary']['medication_count'],
                'conditions': medical_analysis['summary']['condition_count']
            })
        }
    
    except Exception as e:
        logger.error(f"Error in document processor: {str(e)}", exc_info=True)
        
        # Send error message to Slack if possible
        if event.get('channel'):
            error_message = f"❌ Error processing document: {str(e)}\n\nPlease try again or contact support if the issue persists."
            send_slack_response(event['channel'], error_message)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }
