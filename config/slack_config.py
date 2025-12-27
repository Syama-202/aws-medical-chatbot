import os
import json
import logging
from typing import Dict, Any, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackConfig:
    """
    Slack app configuration and client management
    """

    def __init__(self, bot_token: str = None, signing_secret: str = None):
        """
        Initialize Slack configuration

        Args:
            bot_token (str): Slack bot token
            signing_secret (str): Slack signing secret
        """
        self.bot_token = bot_token or os.getenv('SLACK_BOT_TOKEN')
        self.signing_secret = signing_secret or os.getenv('SLACK_SIGNING_SECRET')

        if not self.bot_token:
            raise ValueError("Slack bot token is required")
        if not self.signing_secret:
            raise ValueError("Slack signing secret is required")

        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Slack client"""
        try:
            self.client = WebClient(token=self.bot_token)
            logger.info("Slack client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Slack client: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """
        Test Slack API connection

        Returns:
            bool: True if connection is successful
        """
        try:
            response = self.client.auth_test()
            logger.info(f"Slack connection test successful. Bot user: {response['user']}")
            return True
        except SlackApiError as e:
            logger.error(f"Slack connection test failed: {e.response['error']}")
            return False

    def send_message(self, channel: str, text: str, blocks: Optional[list] = None) -> Dict:
        """
        Send message to Slack channel

        Args:
            channel (str): Channel ID or name
            text (str): Message text
            blocks (list): Message blocks for rich formatting

        Returns:
            Dict: API response
        """
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks
            )
            logger.info(f"Message sent to channel {channel}")
            return response
        except SlackApiError as e:
            logger.error(f"Failed to send message: {e.response['error']}")
            return {"error": e.response['error']}

    def upload_file(self, channels: str, file_path: str, title: str = None) -> Dict:
        """
        Upload file to Slack

        Args:
            channels (str): Channel ID or name
            file_path (str): Path to file
            title (str): File title

        Returns:
            Dict: API response
        """
        try:
            response = self.client.files_upload(
                channels=channels,
                file=file_path,
                title=title
            )
            logger.info(f"File uploaded to channel {channels}")
            return response
        except SlackApiError as e:
            logger.error(f"Failed to upload file: {e.response['error']}")
            return {"error": e.response['error']}

    def get_file_info(self, file_id: str) -> Dict:
        """
        Get file information

        Args:
            file_id (str): Slack file ID

        Returns:
            Dict: File information
        """
        try:
            response = self.client.files_info(file=file_id)
            return response
        except SlackApiError as e:
            logger.error(f"Failed to get file info: {e.response['error']}")
            return {"error": e.response['error']}

    def create_medical_response_blocks(self, 
                                      title: str, 
                                      analysis: str, 
                                      recommendations: list = None,
                                      warnings: list = None) -> list:
        """
        Create formatted blocks for medical responses

        Args:
            title (str): Response title
            analysis (str): Medical analysis text
            recommendations (list): List of recommendations
            warnings (list): List of warnings

        Returns:
            list: Formatted message blocks
        """
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Medical Analysis:*\n{analysis}"
                }
            }
        ]

        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Recommendations:*\n{rec_text}"
                }
            })

        if warnings:
            warn_text = "\n".join([f"⚠️ {warn}" for warn in warnings])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Important Warnings:*\n{warn_text}"
                }
            })

        # Add disclaimer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "_This analysis is for informational purposes only and should not replace professional medical advice._"
                }
            ]
        })

        return blocks

    def format_prescription_analysis(self, analysis_data: Dict) -> Dict:
        """
        Format prescription analysis for Slack display

        Args:
            analysis_data (Dict): Analysis results from Comprehend Medical

        Returns:
            Dict: Formatted message data
        """
        try:
            entities = analysis_data.get('entities', {}).get('Entities', [])

            medications = []
            conditions = []
            dosages = []

            for entity in entities:
                category = entity.get('Category', '')
                text = entity.get('Text', '')

                if category == 'MEDICATION':
                    medications.append(text)
                elif category == 'MEDICAL_CONDITION':
                    conditions.append(text)
                elif category == 'DOSAGE':
                    dosages.append(text)

            # Create summary text
            summary_parts = []

            if medications:
                summary_parts.append(f"**Medications Found:** {', '.join(set(medications))}")

            if conditions:
                summary_parts.append(f"**Conditions:** {', '.join(set(conditions))}")

            if dosages:
                summary_parts.append(f"**Dosages:** {', '.join(set(dosages))}")

            summary = "\n".join(summary_parts) if summary_parts else "No medical entities detected."

            blocks = self.create_medical_response_blocks(
                title="📋 Prescription Analysis Complete",
                analysis=summary,
                warnings=[
                    "Always verify medication details with your healthcare provider",
                    "Check for drug interactions and allergies",
                    "Follow prescribed dosages exactly"
                ]
            )

            return {
                "text": "Prescription analysis completed",
                "blocks": blocks
            }

        except Exception as e:
            logger.error(f"Failed to format prescription analysis: {str(e)}")
            return {
                "text": f"Error formatting analysis: {str(e)}",
                "blocks": []
            }

# Slack configuration settings
SLACK_CONFIG = {
    "bot_token": os.getenv('SLACK_BOT_TOKEN'),
    "signing_secret": os.getenv('SLACK_SIGNING_SECRET'),
    "bot_user_oauth_access_token": os.getenv('SLACK_BOT_USER_OAUTH_ACCESS_TOKEN'),
    "verification_token": os.getenv('SLACK_VERIFICATION_TOKEN'),
    "app_mention_event": "app_mention",
    "message_event": "message",
    "file_share_event": "file_share"
}

# Event handlers configuration
EVENT_HANDLERS = {
    "app_mention": "handle_app_mention",
    "message": "handle_message", 
    "file_share": "handle_file_share",
    "file_created": "handle_file_upload"
}

def initialize_slack_client() -> SlackConfig:
    """
    Initialize Slack client for medical chatbot

    Returns:
        SlackConfig: Configured Slack client
    """
    try:
        slack_client = SlackConfig(
            bot_token=SLACK_CONFIG["bot_token"],
            signing_secret=SLACK_CONFIG["signing_secret"]
        )

        # Test connection
        if slack_client.test_connection():
            logger.info("Slack client initialized and tested successfully")
            return slack_client
        else:
            raise Exception("Slack connection test failed")

    except Exception as e:
        logger.error(f"Failed to initialize Slack client: {str(e)}")
        raise

def create_welcome_message() -> Dict:
    """Create welcome message for new users"""
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🏥 Medical Chatbot Assistant"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Welcome to your AI-powered medical assistant!*\n\nI can help you with:"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "• 📋 Analyze medical prescriptions\n• 🔍 Answer medical questions\n• 📊 Process medical documents\n• 💊 Identify medications and dosages"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*To get started:*\n- Ask me a medical question\n- Upload a prescription or medical document\n- Mention me in a channel with @MedicalBot"
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "_Remember: This bot provides informational assistance only and should not replace professional medical advice._"
                }
            ]
        }
    ]

    return {
        "text": "Welcome to Medical Chatbot Assistant!",
        "blocks": blocks
    }

if __name__ == "__main__":
    # Initialize Slack client
    slack_client = initialize_slack_client()
    print("Slack configuration initialized successfully!")