import json
import logging
import os
import boto3
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import hashlib
import hmac
import time
from urllib.parse import parse_qs

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')
lex_client = boto3.client('lexv2-runtime')

# Environment variables
SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET']
SLACK_BOT_TOKEN = os.environ['SLACK_BOT_TOKEN']
S3_BUCKET = os.environ['S3_BUCKET']
LEX_BOT_ID = os.environ.get('LEX_BOT_ID', '')
LEX_BOT_ALIAS_ID = os.environ.get('LEX_BOT_ALIAS_ID', 'TSTALIASID')
LEX_LOCALE_ID = os.environ.get('LEX_LOCALE_ID', 'en_US')
DOCUMENT_PROCESSOR_FUNCTION = os.environ['DOCUMENT_PROCESSOR_FUNCTION']

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN)

def verify_slack_signature(event):
    """Verify Slack request signature"""
    try:
        timestamp = event['headers'].get('X-Slack-Request-Timestamp', '')
        signature = event['headers'].get('X-Slack-Signature', '')
        body = event.get('body', '')

        # Create the signature base string
        sig_basestring = f"v0:{timestamp}:{body}"

        # Generate the signature
        expected_signature = 'v0=' + hmac.new(
            SLACK_SIGNING_SECRET.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()

        # Verify the signature
        return hmac.compare_digest(signature, expected_signature)

    except Exception as e:
        logger.error(f"Error verifying Slack signature: {str(e)}")
        return False

def handle_slack_event(event_data):
    """Handle Slack events"""
    try:
        event_type = event_data.get('type')

        # URL verification is now handled in lambda_handler
        # This is kept for backward compatibility
        if event_type == 'url_verification':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'text/plain'
                },
                'body': event_data.get('challenge', '')
            }

        elif event_type == 'event_callback':
            # Handle actual events
            event = event_data.get('event', {})
            return handle_event(event)

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'ok'})
        }

    except Exception as e:
        logger.error(f"Error handling Slack event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_event(event):
    """Handle specific event types"""
    try:
        event_type = event.get('type')

        if event_type == 'message':
            return handle_message(event)
        elif event_type == 'file_share':
            return handle_file_share(event)
        elif event_type == 'app_mention':
            return handle_app_mention(event)

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'ok'})
        }

    except Exception as e:
        logger.error(f"Error handling event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_message(event):
    """Handle message events"""
    try:
        # Skip bot messages
        if event.get('bot_id') or event.get('subtype') == 'bot_message':
            return {'statusCode': 200, 'body': json.dumps({'status': 'ignored'})}

        user = event.get('user')
        channel = event.get('channel')
        text = event.get('text', '')

        # Check if message contains files
        if 'files' in event:
            return handle_file_upload(event)

        # Process text message with Lex
        if text.strip():
            response = process_with_lex(text, user, channel)
            send_response_to_slack(channel, response)

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'processed'})
        }

    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_file_share(event):
    """Handle file share events"""
    return handle_file_upload(event)

def handle_file_upload(event):
    """Handle file upload events"""
    try:
        files = event.get('files', [])
        channel = event.get('channel')
        user = event.get('user')

        if not files:
            send_response_to_slack(channel, "No files detected in the message.")
            return {'statusCode': 200, 'body': json.dumps({'status': 'no_files'})}

        for file_info in files:
            file_id = file_info.get('id')
            file_name = file_info.get('name', 'unknown')
            file_type = file_info.get('mimetype', '')

            # Check if it's a supported file type
            if is_supported_file_type(file_type):
                # Process the document
                process_document(file_id, file_name, channel, user)
            else:
                send_response_to_slack(
                    channel,
                    f"Sorry, I don't support {file_type} files. Please upload PDF, PNG, JPG, or JPEG files."
                )

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'files_processed'})
        }

    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        send_response_to_slack(
            event.get('channel', ''),
            f"Error processing your file: {str(e)}"
        )
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_app_mention(event):
    """Handle app mention events"""
    try:
        text = event.get('text', '')
        channel = event.get('channel')
        user = event.get('user')

        # Remove bot mention from text
        text = text.replace(f'<@{slack_client.auth_test()["user_id"]}>', '').strip()

        if text:
            response = process_with_lex(text, user, channel)
            send_response_to_slack(channel, response)
        else:
            send_help_message(channel)

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'mention_processed'})
        }

    except Exception as e:
        logger.error(f"Error handling app mention: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def is_supported_file_type(file_type):
    """Check if file type is supported"""
    supported_types = [
        'application/pdf',
        'image/png',
        'image/jpeg',
        'image/jpg'
    ]
    return file_type.lower() in supported_types

def process_document(file_id, file_name, channel, user):
    """Process uploaded document"""
    try:
        # Send processing message
        send_response_to_slack(
            channel,
            f"📋 Processing your document: {file_name}. Please wait..."
        )

        # Invoke document processor Lambda
        payload = {
            'file_id': file_id,
            'file_name': file_name,
            'channel': channel,
            'user': user
        }

        lambda_client.invoke(
            FunctionName=DOCUMENT_PROCESSOR_FUNCTION,
            InvocationType='Event',  # Asynchronous
            Payload=json.dumps(payload)
        )

        logger.info(f"Document processing initiated for file: {file_name}")

    except Exception as e:
        logger.error(f"Error initiating document processing: {str(e)}")
        send_response_to_slack(
            channel,
            f"Error processing document: {str(e)}"
        )

def process_with_lex(text, user, channel):
    """Process text with Amazon Lex"""
    try:
        if not LEX_BOT_ID:
            return "I'm not configured with a Lex bot. Please set up the bot configuration."

        # Create session ID
        session_id = f"{user}{channel}{int(time.time())}"

        response = lex_client.recognize_text(
            botId=LEX_BOT_ID,
            botAliasId=LEX_BOT_ALIAS_ID,
            localeId=LEX_LOCALE_ID,
            sessionId=session_id,
            text=text
        )

        # Extract response message
        messages = response.get('messages', [])
        if messages:
            return messages[0].get('content', 'I understand your query, but I cannot provide a response at the moment.')
        else:
            return 'I received your message but cannot provide a response at the moment.'

    except Exception as e:
        logger.error(f"Error processing with Lex: {str(e)}")
        return f"I'm having trouble understanding your request. Please try again or contact support."

def send_response_to_slack(channel, message):
    """Send response message to Slack"""
    try:
        # Format message with medical disclaimer
        formatted_message = f"{message}\n\n_⚠ This information is for educational purposes only and should not replace professional medical advice._"

        slack_client.chat_postMessage(
            channel=channel,
            text=formatted_message,
            mrkdwn=True
        )

        logger.info(f"Response sent to channel: {channel}")

    except SlackApiError as e:
        logger.error(f"Error sending message to Slack: {e.response['error']}")
    except Exception as e:
        logger.error(f"Unexpected error sending message: {str(e)}")

def send_help_message(channel):
    """Send help message to Slack"""
    help_text = """
🏥 *Medical Chatbot Assistant Help*

I can help you with:
• 📋 Analyze medical prescriptions and documents
• 💊 Identify medications and dosages  
• 🔍 Answer medical questions
• 📊 Process medical records

*How to use:*
• Ask me a question: "What is diabetes?"
• Upload a prescription or medical document
• Mention me: @MedicalBot what does this prescription contain?

*Supported file types:* PDF, PNG, JPG, JPEG

⚠ Remember: I provide informational assistance only and cannot replace professional medical advice.
    """

    send_response_to_slack(channel, help_text)

def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        logger.info(f"Received event: {json.dumps(event)}")

        # Handle different types of requests
        if event.get('httpMethod'):
            # HTTP API Gateway request
            
            # Parse the request body first
            if event.get('isBase64Encoded'):
                import base64
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event.get('body', '')

            # Parse JSON body for event subscriptions
            try:
                event_data = json.loads(body)
            except json.JSONDecodeError:
                # Not JSON, might be form-encoded (slash command or interactive component)
                event_data = None

            # CRITICAL: Handle URL verification FIRST, before signature verification
            if event_data and event_data.get('type') == 'url_verification':
                logger.info("Handling URL verification challenge")
                challenge = event_data.get('challenge', '')
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'text/plain'
                    },
                    'body': challenge  # Return the challenge value directly as plain text
                }

            # Now verify signature for all other requests
            if not verify_slack_signature(event):
                logger.warning("Invalid Slack signature")
                return {
                    'statusCode': 401,
                    'body': json.dumps({'error': 'Invalid signature'})
                }

            # Handle different request types
            if 'payload=' in body:
                # Interactive component
                payload = json.loads(parse_qs(body)['payload'][0])
                return handle_interactive_component(payload)
            elif body.startswith('token='):
                # Slash command
                parsed_body = parse_qs(body)
                return handle_slash_command(parsed_body)
            elif event_data:
                # Event subscription (after URL verification)
                return handle_slack_event(event_data)
            else:
                logger.warning("Unknown request format")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Unknown request format'})
                }

        else:
            # Direct Lambda invocation
            return handle_slack_event(event)

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'type': 'lambda_handler_error'
            })
        }
    
def handle_slash_command(parsed_body):
    """Handle Slack slash commands"""
    try:
        command = parsed_body.get('command', [''])[0]
        text = parsed_body.get('text', [''])[0]
        channel_id = parsed_body.get('channel_id', [''])[0]
        user_id = parsed_body.get('user_id', [''])[0]

        if command == '/medical':
            if text.strip():
                response = process_with_lex(text, user_id, channel_id)
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'response_type': 'in_channel',
                        'text': response
                    })
                }
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'response_type': 'ephemeral',
                        'text': 'Please provide a medical query. Example: /medical What is diabetes?'
                    })
                }

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response_type': 'ephemeral',
                'text': 'Unknown command'
            })
        }

    except Exception as e:
        logger.error(f"Error handling slash command: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_interactive_component(payload):
    """Handle Slack interactive components"""
    try:
        # Handle button clicks, menu selections, etc.
        action_id = payload.get('actions', [{}])[0].get('action_id', '')

        if action_id == 'get_help':
            channel = payload.get('channel', {}).get('id', '')
            send_help_message(channel)

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'ok'})
        }

    except Exception as e:
        logger.error(f"Error handling interactive component: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }