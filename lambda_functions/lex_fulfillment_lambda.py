import json
import logging
import os
import hashlib
from typing import Dict, Optional, Tuple
from functools import lru_cache
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Constants
MIN_QUERY_LENGTH = 2
MIN_RESPONSE_LENGTH = 50
MAX_INPUT_LENGTH = 500
BEDROCK_TIMEOUT = 30
CACHE_SIZE = 100

# Response templates
MEDICAL_DISCLAIMER = "\n\n⚠️ This is educational information only. Always consult a qualified healthcare professional for medical advice."
MEDICATION_DISCLAIMER = "\n\n⚠️ This is general information only. Always follow your healthcare provider's instructions and consult your pharmacist for medication-specific questions."
SYMPTOM_DISCLAIMER = "\n\n⚠️ IMPORTANT: This is NOT a medical diagnosis. If symptoms are severe, worsening, or concerning, seek immediate medical attention."

# Environment variables
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'amazon.titan-text-express-v1')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'true').lower() == 'true'

# Initialize Bedrock client with retry configuration
bedrock_config = Config(
    region_name=AWS_REGION,
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=5,
    read_timeout=BEDROCK_TIMEOUT
)

try:
    bedrock_runtime = boto3.client('bedrock-runtime', config=bedrock_config)
    logger.info(f"Bedrock runtime client initialized - Region: {AWS_REGION}, Model: {BEDROCK_MODEL_ID}")
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {str(e)}")
    bedrock_runtime = None


class InputValidator:
    """Validates and sanitizes user input"""
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Remove potentially harmful characters"""
        if not text:
            return ""
        # Remove control characters and excessive whitespace
        sanitized = ' '.join(text.split())
        return sanitized[:MAX_INPUT_LENGTH]
    
    @staticmethod
    def validate_query(text: str) -> Tuple[bool, str]:
        """Validate query meets minimum requirements"""
        if not text:
            return False, "Please provide a query."
        
        sanitized = InputValidator.sanitize_input(text)
        
        if len(sanitized) < MIN_QUERY_LENGTH:
            return False, f"Query too short. Please provide more details."
        
        if len(sanitized) > MAX_INPUT_LENGTH:
            return False, f"Query too long. Please keep it under {MAX_INPUT_LENGTH} characters."
        
        return True, sanitized


class PromptBuilder:
    """Builds structured prompts for Bedrock"""
    
    @staticmethod
    def medical_query(condition: str) -> str:
        return f"""Provide clear, concise medical information about: {condition}

Structure your response with:
1. Brief definition (2-3 sentences)
2. Main causes or risk factors (2-3 bullet points)
3. Common symptoms (2-3 bullet points)
4. General treatment approach (1-2 sentences)
5. When to see a doctor (1-2 sentences)

Keep response under 250 words, use patient-friendly language, be factual and helpful.

Medical condition: {condition}

Response:"""

    @staticmethod
    def medication_query(medication: str) -> str:
        return f"""Provide clear, concise information about the medication: {medication}

Structure your response with:
1. What it treats (1-2 sentences)
2. How it works (1-2 sentences)
3. Common forms (tablets, liquid, etc.) (1 sentence)
4. Common side effects (3-4 bullet points)
5. Important warnings (1-2 sentences)

Keep response under 250 words, use patient-friendly language.

Medication: {medication}

Response:"""

    @staticmethod
    def symptom_checker(symptoms: str) -> str:
        return f"""A patient reports these symptoms: {symptoms}

Provide empathetic, helpful information:
1. Acknowledge their symptoms (1 sentence)
2. Possible common causes - educational only, not diagnosis (2-3 bullet points)
3. Self-care tips if appropriate (2-3 bullet points)
4. When to seek medical care (2-3 bullet points)
5. Emergency warning signs if relevant (1-2 bullet points)

Keep response under 250 words, use supportive tone, emphasize this is NOT a diagnosis.

Symptoms: {symptoms}

Response:"""

    @staticmethod
    def general_query(query: str) -> str:
        return f"""User asked: "{query}"

Provide a brief, helpful response if this is health-related. If not health-related, politely redirect to medical topics.

Keep response under 150 words.

Response:"""


class BedrockClient:
    """Handles all Bedrock API interactions"""
    
    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id
        self.cache = {}
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def invoke(self, prompt: str, use_cache: bool = True) -> Optional[str]:
        """Call Bedrock model with optional caching"""
        
        if not self.client:
            logger.error("Bedrock client not initialized")
            return None
        
        # Check cache
        if use_cache and ENABLE_CACHING:
            cache_key = self._generate_cache_key(prompt)
            if cache_key in self.cache:
                logger.info("Cache hit - returning cached response")
                return self.cache[cache_key]
        
        try:
            logger.info(f"Calling Bedrock - Model: {self.model_id}, Prompt length: {len(prompt)}")
            
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1024,
                    "temperature": 0.2,
                    "topP": 0.9,
                    "stopSequences": []
                }
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            results = response_body.get('results', [])
            
            if results and len(results) > 0:
                answer = results[0].get('outputText', '').strip()
                logger.info(f"Bedrock response received: {len(answer)} characters")
                
                # Cache the response
                if use_cache and ENABLE_CACHING and len(self.cache) < CACHE_SIZE:
                    cache_key = self._generate_cache_key(prompt)
                    self.cache[cache_key] = answer
                
                return answer
            else:
                logger.error(f"No results in Bedrock response: {response_body}")
                return None
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', 'Unknown')
            logger.error(f"Bedrock ClientError - Code: {error_code}, Message: {error_message}")
            return None
        except Exception as e:
            logger.error(f"Error calling Bedrock: {str(e)}", exc_info=True)
            return None


# Initialize Bedrock client wrapper
bedrock_client = BedrockClient(bedrock_runtime, BEDROCK_MODEL_ID) if bedrock_runtime else None


def lambda_handler(event, context):
    """
    Amazon Lex fulfillment Lambda function
    Handles intent fulfillment for the medical chatbot
    """
    intent_name = 'MedicalQuery'  # Default
    slots = {}
    
    try:
        logger.info(f"=== NEW REQUEST ===")
        logger.debug(f"Lex event: {json.dumps(event)}")
        
        # Extract intent details
        session_state = event.get('sessionState', {})
        intent = session_state.get('intent', {})
        intent_name = intent.get('name', 'MedicalQuery')
        slots = intent.get('slots', {})
        
        input_transcript = event.get('inputTranscript', '')
        session_id = event.get('sessionId', 'unknown')
        
        logger.info(f"Session: {session_id} | Intent: {intent_name} | Input: {input_transcript[:100]}")
        
        # Route to appropriate handler
        handlers = {
            'MedicalQuery': handle_medical_query,
            'MedicationQuery': handle_medication_query,
            'AnalyzePrescription': handle_prescription_analysis,
            'SymptomChecker': handle_symptom_checker,
            'GetHelp': handle_help_request,
            'FallbackIntent': handle_fallback
        }
        
        handler = handlers.get(intent_name, handle_unknown_intent)
        return handler(event, input_transcript, slots, intent_name)
    
    except Exception as e:
        logger.error(f"Critical error in lambda_handler: {str(e)}", exc_info=True)
        return create_lex_response(
            "I apologize, but I encountered a technical error. Please try again in a moment.",
            'Failed',
            intent_name=intent_name,
            slots=slots
        )


def handle_medical_query(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle general medical queries using Bedrock"""
    try:
        # Validate input
        is_valid, result = InputValidator.validate_query(input_transcript)
        if not is_valid:
            return create_lex_response(
                "I'd be happy to provide medical information. What medical condition would you like to learn about?",
                'Fulfilled',
                intent_name=intent_name,
                slots=slots
            )
        
        condition = result
        logger.info(f"Processing medical query: {condition}")
        
        # Get AI response
        if bedrock_client:
            prompt = PromptBuilder.medical_query(condition)
            answer = bedrock_client.invoke(prompt)
            
            if answer and len(answer) >= MIN_RESPONSE_LENGTH:
                final_answer = f"{answer}{MEDICAL_DISCLAIMER}"
                logger.info("Successfully generated medical answer")
                return create_lex_response(final_answer, 'Fulfilled', intent_name=intent_name, slots=slots)
        
        # Fallback response
        logger.warning("Using fallback response for medical query")
        fallback = get_medical_fallback(condition)
        return create_lex_response(fallback, 'Fulfilled', intent_name=intent_name, slots=slots)
    
    except Exception as e:
        logger.error(f"Error in handle_medical_query: {str(e)}", exc_info=True)
        fallback = get_medical_fallback(input_transcript)
        return create_lex_response(fallback, 'Fulfilled', intent_name=intent_name, slots=slots)


def handle_medication_query(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle medication-specific queries using Bedrock"""
    try:
        # Validate input
        is_valid, result = InputValidator.validate_query(input_transcript)
        if not is_valid:
            return create_lex_response(
                "I can provide information about medications. Which medication would you like to know about?",
                'Fulfilled',
                intent_name=intent_name,
                slots=slots
            )
        
        medication = result
        logger.info(f"Processing medication query: {medication}")
        
        # Get AI response
        if bedrock_client:
            prompt = PromptBuilder.medication_query(medication)
            answer = bedrock_client.invoke(prompt)
            
            if answer and len(answer) >= MIN_RESPONSE_LENGTH:
                final_answer = f"{answer}{MEDICATION_DISCLAIMER}"
                logger.info("Successfully generated medication answer")
                return create_lex_response(final_answer, 'Fulfilled', intent_name=intent_name, slots=slots)
        
        # Fallback response
        logger.warning("Using fallback response for medication query")
        fallback = get_medication_fallback(medication)
        return create_lex_response(fallback, 'Fulfilled', intent_name=intent_name, slots=slots)
    
    except Exception as e:
        logger.error(f"Error in handle_medication_query: {str(e)}", exc_info=True)
        fallback = get_medication_fallback(input_transcript)
        return create_lex_response(fallback, 'Fulfilled', intent_name=intent_name, slots=slots)


def handle_prescription_analysis(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle prescription analysis requests"""
    try:
        logger.info("Handling prescription analysis request")
        response_text = """I can help analyze prescription documents! Please upload your prescription by:

1. Click the attachment/paperclip icon in your chat application
2. Select your prescription file (PDF, PNG, or JPG)
3. Add a message like "Analyze this prescription"

I'll extract the medications, dosages, and provide important safety information.

⚠️ Always verify prescription details with your pharmacist or healthcare provider."""

        return create_lex_response(response_text, 'Fulfilled', intent_name=intent_name, slots=slots)
    except Exception as e:
        logger.error(f"Error in handle_prescription_analysis: {str(e)}", exc_info=True)
        return create_lex_response(
            "I can help analyze prescription documents when you upload them. Please try uploading a file.",
            'Fulfilled',
            intent_name=intent_name,
            slots=slots
        )


def handle_symptom_checker(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle symptom checking requests using Bedrock"""
    try:
        # Validate input
        is_valid, result = InputValidator.validate_query(input_transcript)
        if not is_valid:
            return create_lex_response(
                "I can help you understand your symptoms. What symptoms are you experiencing?",
                'Fulfilled',
                intent_name=intent_name,
                slots=slots
            )
        
        symptoms = result
        logger.info(f"Processing symptom check: {symptoms}")
        
        # Get AI response
        if bedrock_client:
            prompt = PromptBuilder.symptom_checker(symptoms)
            answer = bedrock_client.invoke(prompt)
            
            if answer and len(answer) >= MIN_RESPONSE_LENGTH:
                final_answer = f"{answer}{SYMPTOM_DISCLAIMER}"
                logger.info("Successfully generated symptom response")
                return create_lex_response(final_answer, 'Fulfilled', intent_name=intent_name, slots=slots)
        
        # Fallback response
        logger.warning("Using fallback response for symptoms")
        fallback = get_symptom_fallback(symptoms)
        return create_lex_response(fallback, 'Fulfilled', intent_name=intent_name, slots=slots)
    
    except Exception as e:
        logger.error(f"Error in handle_symptom_checker: {str(e)}", exc_info=True)
        fallback = get_symptom_fallback(input_transcript)
        return create_lex_response(fallback, 'Fulfilled', intent_name=intent_name, slots=slots)


def handle_help_request(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle help requests"""
    logger.info("Handling help request")
    
    help_text = """Medical Chatbot Help

I can assist you with:

Medical Information
Ask me about medical conditions, diseases, or health topics.
Example: "What is diabetes?" or "Tell me about high blood pressure"

Medication Information
Learn about medications, their uses, and side effects.
Example: "What is aspirin?" or "Tell me about paracetamol"

Symptom Checker
Describe your symptoms for general health guidance.
Example: "I have a headache and fever" or "My throat hurts"

Prescription Analysis
Upload prescription documents for analysis.

⚠️ Important: I provide educational information only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment."""

    return create_lex_response(help_text, 'Fulfilled', intent_name=intent_name, slots=slots)


def handle_fallback(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle unrecognized inputs"""
    logger.info(f"Handling fallback for input: {input_transcript}")
    
    # Try Bedrock for unknown queries
    if bedrock_client and len(input_transcript) > 5:
        try:
            prompt = PromptBuilder.general_query(input_transcript)
            answer = bedrock_client.invoke(prompt)
            
            if answer and len(answer) > 30:
                final_answer = f"{answer}\n\n💡 Tip: Ask me about medical conditions, medications, or symptoms!"
                return create_lex_response(final_answer, 'Fulfilled', intent_name=intent_name, slots=slots)
        except Exception as e:
            logger.error(f"Error in fallback Bedrock call: {str(e)}")
    
    # Static fallback
    fallback_messages = [
        "I can help with medical questions, medications, and symptoms. What would you like to know?",
        f"I'm not sure how to help with that. Try asking about medical conditions, medications, or symptoms.",
        "Ask me about medical conditions, medications, or describe your symptoms!"
    ]
    
    # Deterministic selection based on input
    index = int(hashlib.md5(input_transcript.encode()).hexdigest(), 16) % len(fallback_messages)
    return create_lex_response(fallback_messages[index], 'Fulfilled', intent_name=intent_name, slots=slots)


def handle_unknown_intent(event, input_transcript: str, slots: Dict, intent_name: str):
    """Handle unknown intents"""
    logger.warning(f"Unknown intent received: {intent_name}")
    return create_lex_response(
        "I don't recognize that request type. Ask me about medical conditions, medications, or symptoms!",
        'Fulfilled',
        intent_name=intent_name,
        slots=slots
    )


# Fallback response generators
def get_medical_fallback(condition: str) -> str:
    return f"""I understand you're asking about {condition}. For comprehensive, accurate medical information about this topic, I recommend:

• Consulting with a healthcare professional
• Visiting reputable medical websites like Mayo Clinic (mayoclinic.org) or WebMD (webmd.com)
• Contacting your doctor's office if you have specific concerns

Is there anything else I can help you with?"""


def get_medication_fallback(medication: str) -> str:
    return f"""For accurate information about {medication}, please:

• Consult your pharmacist - they can provide detailed medication information
• Speak with your healthcare provider about this medication
• Check the medication package insert for official information
• Visit FDA.gov for approved drug information

Can I help you with anything else?"""


def get_symptom_fallback(symptoms: str) -> str:
    return f"""You mentioned: {symptoms}

General Guidance:

Seek medical care if your symptoms:
• Are severe or rapidly worsening
• Persist for more than a few days
• Include fever, chest pain, or breathing difficulty
• Significantly affect daily activities
• Cause you concern

🚨 EMERGENCY: Call emergency services immediately if experiencing:
• Severe chest pain or pressure
• Difficulty breathing or shortness of breath
• Severe bleeding
• Loss of consciousness
• Stroke symptoms (face drooping, arm weakness, speech difficulty)

⚠️ This is general information only, NOT a medical diagnosis. Consult a healthcare professional for proper evaluation."""


def create_lex_response(message: str, fulfillment_state: str, slot_to_elicit: Optional[str] = None, 
                       intent_name: str = 'MedicalQuery', slots: Dict = None) -> Dict:
    """Create properly formatted Lex V2 response"""
    if slots is None:
        slots = {}
    
    logger.info(f"Creating response - Intent: {intent_name}, State: {fulfillment_state}")
    
    response = {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitSlot' if slot_to_elicit else 'Close',
            },
            'intent': {
                'name': intent_name,
                'slots': slots,
                'state': 'InProgress' if slot_to_elicit else fulfillment_state
            }
        },
        'messages': [
            {
                'contentType': 'PlainText',
                'content': message
            }
        ]
    }
    
    if slot_to_elicit:
        response['sessionState']['dialogAction']['slotToElicit'] = slot_to_elicit
    
    return response