@echo off
REM Medical Bedrock Chatbot - Windows Development Setup

echo 🏥 Setting up Medical Bedrock Chatbot Development Environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [INFO] Python is installed

REM Create virtual environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [WARNING] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [INFO] Installing Python dependencies...
pip install -r requirements.txt
echo [SUCCESS] Dependencies installed

REM Install development dependencies  
echo [INFO] Installing development dependencies...
pip install -r requirements-dev.txt
echo [SUCCESS] Development dependencies installed

REM Create .env file from example
if not exist ".env" (
    echo [INFO] Creating .env file from template...
    copy .env.example .env
    echo [WARNING] Please edit .env file with your actual credentials
) else (
    echo [WARNING] .env file already exists
)

REM Check AWS CLI installation
aws --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] AWS CLI is installed
) else (
    echo [WARNING] AWS CLI not found. Please install it for deployment.
    echo Installation: https://aws.amazon.com/cli/
)

REM Create necessary directories
echo [INFO] Creating project directories...
mkdir lambda_functions\slack_handler lambda_functions\document_processor lambda_functions\rag_processor lambda_functions\lex_fulfillment >nul 2>&1
mkdir config utils tests\unit tests\integration tests\fixtures >nul 2>&1
mkdir docs\images scripts infrastructure\cloudformation infrastructure\terraform >nul 2>&1
mkdir local_dev\docker local_dev\mock_services >nul 2>&1
echo [SUCCESS] Project directories created

REM Install pre-commit hooks
echo [INFO] Setting up pre-commit hooks...
pre-commit install
echo [SUCCESS] Pre-commit hooks installed

echo [SUCCESS] 🎉 Local development environment setup complete!
echo.
echo [INFO] Next steps:
echo 1. Edit .env file with your credentials
echo 2. Configure AWS credentials: aws configure  
echo 3. Run: python config\pinecone_setup.py
echo 4. Run: python infrastructure\deploy.py
echo.
echo [INFO] To activate virtual environment: venv\Scripts\activate.bat
echo [INFO] To run tests: python -m pytest tests\

pause
