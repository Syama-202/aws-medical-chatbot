import json
import logging
import os
import boto3
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from slack_sdk import WebClient
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
s3_client = boto3.client('s3')

# Environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'medical-docs')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', 'us-east-1-aws')
SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN', '')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'amazon.titan-text-express-v1')
BEDROCK_EMBEDDING_MODEL_ID = os.environ.get('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')
VECTOR_DIMENSION = 1536  # Titan embedding dimension

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None

# Initialize Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
    pinecone_index = None
    
    if pc:
        # Initialize or get index
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("Pinecone initialized successfully")
except ImportError as e:
    logger.error(f"Pinecone library not available: {str(e)}")
    pc = None
    pinecone_index = None
except Exception as e:
    logger.error(f"Error initializing Pinecone: {str(e)}")
    pc = None
    pinecone_index = None

# ==================== EMBEDDING FUNCTIONS ====================

def create_text_embeddings(text: str) -> List[float]:
    """
    Generate embeddings using Amazon Titan Embed model
    Returns a 1536-dimensional vector
    """
    try:
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding")
            return [0.0] * VECTOR_DIMENSION
        
        # Truncate text if too long (Titan has token limits)
        max_chars = 8000
        text = text[:max_chars] if len(text) > max_chars else text
        
        body = {
            "inputText": text
        }
        
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])
        
        if not embedding or len(embedding) != VECTOR_DIMENSION:
            logger.error(f"Invalid embedding dimension: {len(embedding)}")
            return [0.0] * VECTOR_DIMENSION
        
        logger.info(f"Generated embedding vector of dimension {len(embedding)}")
        return embedding
        
    except ClientError as e:
        logger.error(f"Bedrock ClientError in embeddings: {str(e)}")
        return [0.0] * VECTOR_DIMENSION
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return [0.0] * VECTOR_DIMENSION

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for period, question mark, or exclamation
            last_period = text[start:end].rfind('.')
            last_question = text[start:end].rfind('?')
            last_exclamation = text[start:end].rfind('!')
            
            boundary = max(last_period, last_question, last_exclamation)
            if boundary > chunk_size // 2:  # Only use if not too far back
                end = start + boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

# ==================== PINECONE STORAGE FUNCTIONS ====================

def store_medical_document_vectors(
    text: str, 
    metadata: Dict,
    namespace: str = "medical_docs"
) -> Tuple[bool, List[str]]:
    """
    Store medical document in Pinecone with intelligent chunking
    Returns (success, list of document IDs)
    """
    try:
        if not pinecone_index:
            logger.warning("Pinecone not initialized, skipping vector storage")
            return False, []
        
        # Chunk the text for better retrieval
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        vectors_to_upsert = []
        doc_ids = []
        base_doc_id = f"doc_{int(time.time())}_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}"
        
        for idx, chunk in enumerate(chunks):
            # Generate embedding for each chunk
            embedding = create_text_embeddings(chunk)
            
            # Create unique ID for each chunk
            chunk_id = f"{base_doc_id}_chunk_{idx}"
            doc_ids.append(chunk_id)
            
            # Prepare metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'chunk_text': chunk[:500],  # Store preview of chunk
                'base_doc_id': base_doc_id,
                'created_at': datetime.utcnow().isoformat()
            })
            
            vectors_to_upsert.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': chunk_metadata
            })
        
        # Batch upsert to Pinecone
        if vectors_to_upsert:
            pinecone_index.upsert(
                vectors=vectors_to_upsert,
                namespace=namespace
            )
            logger.info(f"Stored {len(vectors_to_upsert)} document chunks in Pinecone")
            return True, doc_ids
        else:
            logger.warning("No vectors to upsert")
            return False, []
            
    except Exception as e:
        logger.error(f"Error storing vectors in Pinecone: {str(e)}")
        return False, []

def query_similar_documents(
    query_text: str,
    top_k: int = 5,
    namespace: str = "medical_docs",
    filter_dict: Optional[Dict] = None,
    score_threshold: float = 0.7
) -> List[Dict]:
    """
    Query Pinecone for similar documents with advanced filtering
    """
    try:
        if not pinecone_index:
            logger.warning("Pinecone not initialized, returning empty results")
            return []
        
        # Generate query embedding
        query_embedding = create_text_embeddings(query_text)
        
        # Build query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True,
            'namespace': namespace
        }
        
        # Add filter if provided
        if filter_dict:
            query_params['filter'] = filter_dict
        
        # Query Pinecone
        results = pinecone_index.query(**query_params)
        
        # Process and filter results
        similar_docs = []
        for match in results.get('matches', []):
            score = match.get('score', 0)
            
            # Filter by score threshold
            if score >= score_threshold:
                similar_docs.append({
                    'id': match['id'],
                    'score': score,
                    'metadata': match.get('metadata', {}),
                    'relevance': 'high' if score >= 0.85 else 'medium'
                })
        
        logger.info(f"Found {len(similar_docs)} similar documents (threshold: {score_threshold})")
        return similar_docs
        
    except Exception as e:
        logger.error(f"Error querying similar documents: {str(e)}")
        return []

def delete_document_vectors(base_doc_id: str, namespace: str = "medical_docs") -> bool:
    """
    Delete all chunks of a document from Pinecone
    """
    try:
        if not pinecone_index:
            logger.warning("Pinecone not initialized")
            return False
        
        # Delete by prefix (all chunks of this document)
        pinecone_index.delete(
            filter={'base_doc_id': base_doc_id},
            namespace=namespace
        )
        
        logger.info(f"Deleted document {base_doc_id} from Pinecone")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting document vectors: {str(e)}")
        return False

# ==================== RAG GENERATION FUNCTIONS ====================

def build_rag_context(similar_docs: List[Dict], max_context_length: int = 2000) -> str:
    """
    Build context from retrieved documents for RAG
    """
    if not similar_docs:
        return ""
    
    context_parts = ["**Relevant Medical Information:**\n"]
    current_length = len(context_parts[0])
    
    for idx, doc in enumerate(similar_docs[:5], 1):  # Limit to top 5
        metadata = doc.get('metadata', {})
        score = doc.get('score', 0)
        relevance = doc.get('relevance', 'medium')
        
        # Extract key information from metadata
        chunk_text = metadata.get('chunk_text', '')
        doc_type = metadata.get('type', 'medical_document')
        
        if chunk_text:
            entry = f"\n{idx}. [{relevance.upper()} RELEVANCE - {score:.2f}] {chunk_text[:300]}...\n"
            
            if current_length + len(entry) > max_context_length:
                break
            
            context_parts.append(entry)
            current_length += len(entry)
    
    if len(context_parts) > 1:
        return "".join(context_parts)
    return ""

def generate_medical_insights(
    text: str,
    medical_analysis: Dict,
    similar_docs: Optional[List[Dict]] = None
) -> str:
    """
    Generate comprehensive medical insights using RAG
    """
    try:
        # Build context from similar documents
        context = build_rag_context(similar_docs) if similar_docs else ""
        
        # Extract key information from medical analysis
        summary = medical_analysis.get('summary', {})
        medications = [med.get('text', '') for med in summary.get('medications', [])][:10]
        conditions = [cond.get('text', '') for cond in summary.get('conditions', [])][:10]
        procedures = [proc.get('text', '') for proc in summary.get('procedures', [])][:10]
        
        # Build comprehensive prompt
        prompt = f"""You are an expert medical AI assistant analyzing a medical document.

**Document Preview:**
{text[:1500]}...

**Extracted Medical Entities:**
- Medications: {', '.join(medications) if medications else 'None detected'}
- Conditions: {', '.join(conditions) if conditions else 'None detected'}
- Procedures: {', '.join(procedures) if procedures else 'None detected'}
- Total entities: {summary.get('total_entities', 0)}

{context}

**Task:** Provide a comprehensive medical analysis including:
1. **Document Summary** (2-3 sentences)
2. **Key Medical Findings** (bullet points)
3. **Medication Information** (if applicable)
4. **Important Warnings or Considerations**
5. **Recommended Follow-up Actions**

**Guidelines:**
- Be clear, concise, and professional
- Highlight critical information
- Always emphasize consulting healthcare professionals
- Use patient-friendly language

**Analysis:**"""

        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 2000,
                "temperature": 0.3,
                "topP": 0.9,
                "stopSequences": []
            }
        }
        
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        generated_text = response_body['results'][0]['outputText'].strip()
        
        logger.info("Medical insights generated successfully")
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating medical insights: {str(e)}")
        return f"⚠️ Unable to generate detailed insights. Please consult with a healthcare professional for proper medical analysis.\n\nError: {str(e)}"

def answer_medical_query(question: str, use_rag: bool = True) -> str:
    """
    Answer medical queries using RAG-enhanced generation
    """
    try:
        logger.info(f"Answering medical query: {question[:100]}...")
        
        context = ""
        if use_rag and pinecone_index:
            # Retrieve similar documents
            similar_docs = query_similar_documents(
                query_text=question,
                top_k=5,
                score_threshold=0.65
            )
            
            if similar_docs:
                context = build_rag_context(similar_docs, max_context_length=1500)
                logger.info(f"Retrieved {len(similar_docs)} relevant documents for context")
        
        # Build prompt
        prompt = f"""{context}

**Patient Question:** {question}

**Task:** Provide a comprehensive, accurate answer to this medical question.

**Include:**
1. Clear explanation (3-4 sentences)
2. Key points to remember (3-4 bullet points)
3. When to seek professional care (2-3 bullet points)
4. Important medical disclaimer

**Guidelines:**
- Be accurate and evidence-based
- Use simple, patient-friendly language
- Emphasize consulting healthcare professionals
- Avoid medical jargon where possible

**Answer:**"""

        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1200,
                "temperature": 0.2,
                "topP": 0.9,
                "stopSequences": []
            }
        }
        
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        generated_text = response_body['results'][0]['outputText'].strip()
        
        logger.info("Medical query answered successfully")
        return generated_text
        
    except Exception as e:
        logger.error(f"Error answering medical query: {str(e)}")
        return f"⚠️ Unable to process your question at this time. Please consult with a qualified healthcare professional.\n\nError: {str(e)}"

# ==================== SLACK INTEGRATION ====================

def send_slack_response(channel: str, message: str, thread_ts: Optional[str] = None):
    """
    Send formatted response to Slack with error handling
    """
    try:
        if not slack_client:
            logger.warning("Slack client not initialized")
            return
        
        # Add medical disclaimer
        full_message = f"{message}\n\n{'─' * 50}\n_⚠️ **Medical Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for medical decisions, diagnosis, or treatment._"
        
        # Send message
        response = slack_client.chat_postMessage(
            channel=channel,
            text=full_message,
            mrkdwn=True,
            thread_ts=thread_ts  # Reply in thread if provided
        )
        
        logger.info(f"Response sent to Slack channel: {channel}")
        return response
        
    except Exception as e:
        logger.error(f"Error sending Slack response: {str(e)}")

# ==================== LAMBDA HANDLER ====================

def lambda_handler(event, context):
    """
    Main Lambda handler for RAG processor
    """
    try:
        logger.info(f"=== RAG PROCESSOR INVOKED ===")
        logger.info(f"Event: {json.dumps(event)}")
        
        action = event.get('action', 'generate_insights')
        
        # ===== GENERATE INSIGHTS ACTION =====
        if action == 'generate_insights':
            extracted_text = event.get('extracted_text', '')
            medical_analysis = event.get('medical_analysis', {})
            channel = event.get('channel')
            user = event.get('user', 'unknown')
            
            if not extracted_text:
                raise ValueError("Missing extracted_text parameter")
            
            if not channel:
                logger.warning("No channel provided, skipping Slack notification")
            
            # Prepare metadata for vector storage
            metadata = {
                'user': user,
                'timestamp': int(time.time()),
                'type': 'prescription_document',
                'entity_count': medical_analysis.get('summary', {}).get('total_entities', 0),
                'has_medications': len(medical_analysis.get('summary', {}).get('medications', [])) > 0,
                'has_conditions': len(medical_analysis.get('summary', {}).get('conditions', [])) > 0
            }
            
            # Store document vectors in Pinecone
            success, doc_ids = store_medical_document_vectors(
                text=extracted_text,
                metadata=metadata,
                namespace="medical_docs"
            )
            
            if success:
                logger.info(f"Stored {len(doc_ids)} document chunks")
            
            # Query similar documents
            similar_docs = query_similar_documents(
                query_text=extracted_text[:500],  # Use first part for similarity
                top_k=5,
                score_threshold=0.7
            )
            
            # Generate insights using RAG
            insights = generate_medical_insights(
                text=extracted_text,
                medical_analysis=medical_analysis,
                similar_docs=similar_docs
            )
            
            # Send to Slack if channel provided
            if channel:
                formatted_message = f"🤖 **AI Medical Document Analysis**\n\n{insights}"
                send_slack_response(channel, formatted_message)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success',
                    'message': 'Insights generated successfully',
                    'similar_docs_found': len(similar_docs),
                    'doc_ids': doc_ids,
                    'insights': insights
                })
            }
        
        # ===== ANSWER QUERY ACTION =====
        elif action == 'answer_query':
            question = event.get('question', '')
            channel = event.get('channel')
            use_rag = event.get('use_rag', True)
            
            if not question:
                raise ValueError("Missing question parameter")
            
            # Generate answer
            answer = answer_medical_query(question, use_rag=use_rag)
            
            # Send to Slack if channel provided
            if channel:
                send_slack_response(channel, answer)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success',
                    'message': json.dumps({'response': answer})
                })
            }
        
        # ===== DELETE DOCUMENT ACTION =====
        elif action == 'delete_document':
            doc_id = event.get('doc_id')
            namespace = event.get('namespace', 'medical_docs')
            
            if not doc_id:
                raise ValueError("Missing doc_id parameter")
            
            success = delete_document_vectors(doc_id, namespace)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success' if success else 'failed',
                    'message': f'Document {doc_id} deletion {"successful" if success else "failed"}'
                })
            }
        
        # ===== UNKNOWN ACTION =====
        else:
            raise ValueError(f"Unknown action: {action}")
            
    except Exception as e:
        logger.error(f"Error in RAG processor: {str(e)}", exc_info=True)
        
        # Send error to Slack if channel provided
        if event.get('channel'):
            error_message = f"🤖 **Error Processing Request**\n\n{str(e)}\n\nPlease try again or contact support."
            send_slack_response(event['channel'], error_message)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }
