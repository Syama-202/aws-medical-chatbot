# Medical Chatbot - AI-Powered Healthcare Assistant

An advanced medical chatbot built with Amazon Bedrock, AWS services, and Slack integration that provides intelligent analysis of medical prescriptions, answers medical questions, and offers healthcare guidance.

## 🏥 Features

- **📋 Prescription Analysis**: Upload and analyze medical prescriptions using OCR and AI
- **💊 Medication Information**: Get detailed information about drugs, dosages, and interactions
- **🔍 Medical Q&A**: Ask questions about symptoms, conditions, and treatments
- **🤖 RAG-Powered Insights**: Retrieval-Augmented Generation for accurate, context-aware responses
- **💬 Slack Integration**: Seamless interaction through Slack workspace
- **🔒 HIPAA-Conscious**: Secure handling of medical information

## 🛠 Technology Stack

- **AI/ML**: Amazon Bedrock (Claude 3.5), Amazon Lex, Comprehend Medical
- **Cloud**: AWS Lambda, S3, Textract, API Gateway
- **Vector DB**: Pinecone for semantic search
- **Integration**: Slack SDK for real-time communication
- **Infrastructure**: Serverless architecture with AWS services

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- AWS Account with appropriate permissions
- Slack workspace with bot permissions
- Pinecone account for vector storage

### Installation

1. **Clone and setup environment**

   ```bash
   git clone <repository-url>
   cd medical-chatbot

   # Linux/Mac
   chmod +x scripts/local_setup.sh
   ./scripts/local_setup.sh

   # Windows
   scripts\local_setup.bat
   ```

2. **Configure environment variables**
   Update `.env` file with your API keys:

   ```env
   SLACK_BOT_TOKEN=xoxb-your-token
   SLACK_SIGNING_SECRET=your-secret
   PINECONE_API_KEY=your-key
   AWS_REGION=us-east-1
   ```

3. **Deploy to AWS**

   ```bash
   python infrastructure/deploy.py
   ```

4. **Configure Slack app**
   - Set up Event Subscriptions URL
   - Configure bot permissions and scopes
   - Install app to your workspace

## 📚 Usage

### Slack Commands

- **Medical Questions**: `@MedicalBot what is diabetes?`
- **Upload Prescriptions**: Upload PDF/image files for analysis
- **Symptom Checker**: `I have a headache and fever`
- **Help**: `@MedicalBot help`

### API Integration

The system also provides REST API endpoints for programmatic access:

```python
import requests

# Analyze medical text
response = requests.post(
    'https://your-api-gateway-url/analyze',
    json={'text': 'Patient has hypertension, prescribed Lisinopril 10mg'}
)
```

## 🏗 Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│    Slack    │───▶│  API Gateway │───▶│   Lambda    │
│             │    │              │    │  Functions  │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
    ┌─────────────────────────────────────────┼─────────────────────────────┐
    │                                         ▼                             │
    │  ┌──────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐   │
    │  │    S3    │  │ Textract │  │ Comprehend  │  │     Bedrock      │   │
    │  │ Storage  │  │   OCR    │  │   Medical   │  │   (Claude 3.5)   │   │
    │  └──────────┘  └──────────┘  └─────────────┘  └──────────────────┘   │
    │                                         │                             │
    └─────────────────────────────────────────┼─────────────────────────────┘
                                              ▼
                                    ┌─────────────┐
                                    │  Pinecone   │
                                    │ Vector DB   │
                                    └─────────────┘
```

## 🔧 Configuration

### AWS Services Setup

1. **IAM Roles**: Configure with necessary permissions for Lambda, S3, Bedrock, etc.
2. **S3 Bucket**: For storing uploaded medical documents
3. **Amazon Lex**: Bot for natural language understanding
4. **Bedrock**: Enable Claude 3.5 model access

### Slack App Configuration

1. Create new Slack app at [api.slack.com](https://api.slack.com)
2. Configure Bot Token Scopes:
   - `channels:read`, `chat:write`, `files:read`
3. Set up Event Subscriptions
4. Install app to workspace

### Pinecone Setup

1. Create index with 1536 dimensions
2. Configure for similarity search
3. Set up API key authentication

## 🧪 Testing

Run the test suite:

```bash
# Verify installation
python test_setup.py

# Run unit tests
pytest tests/

# Test individual components
python -m lambda_functions.document_processor
```

## 📋 API Documentation

### Endpoints

- `POST /slack/events` - Slack event handling
- `POST /analyze/prescription` - Direct prescription analysis
- `GET /health` - Health check endpoint

### Response Formats

```json
{
  "status": "success",
  "analysis": {
    "medications": ["Lisinopril 10mg", "Metformin 500mg"],
    "conditions": ["Hypertension", "Diabetes Type 2"],
    "confidence": 0.95
  },
  "recommendations": [
    "Monitor blood pressure regularly",
    "Check blood glucose levels"
  ]
}
```

## 🔒 Security & Compliance

- **HIPAA Considerations**: No PHI stored permanently
- **Encryption**: All data encrypted in transit and at rest
- **Access Control**: IAM roles with least privilege
- **Audit Logging**: CloudWatch for all operations

## 🐛 Troubleshooting

### Common Issues

1. **Slack events not received**

   - Verify Event Subscriptions URL
   - Check API Gateway logs

2. **Textract extraction fails**

   - Ensure clear, high-quality images
   - Check supported file formats

3. **Pinecone connection issues**
   - Verify API key and index name
   - Check network connectivity

### Logs and Monitoring

```bash
# View Lambda logs
aws logs tail /aws/lambda/medical-chatbot-slack-handler

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics --namespace AWS/Lambda
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

This chatbot provides educational information only and should not replace professional medical advice. Always consult qualified healthcare providers for medical decisions and treatment.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Email**: support@medicalchatbot.com

---

Built with ❤️ for healthcare professionals and patients seeking reliable medical information.
