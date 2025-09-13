#!/usr/bin/env python3
"""
RAG Chatbot Startup Script
Handles initial setup and configuration validation
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {platform.python_version()}")
        return False
    print(f"✅ Python {platform.python_version()}")
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  Not running in virtual environment (recommended)")
        response = input("Continue anyway? (y/N): ").lower()
        return response == 'y'

def install_requirements():
    """Install required packages"""
    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📄 Creating .env file from template...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ .env file created")
            print("🔧 Please edit .env file with your API keys and credentials")
            return False  # Need configuration
        else:
            # Create basic .env file
            with open('.env', 'w') as f:
                f.write("""# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id-here
GOOGLE_CLIENT_SECRET=your-google-client-secret-here

# LLM API Configuration (choose one)
GEMINI_API_KEY=your-gemini-api-key-here
# HF_API_KEY=your-hugging-face-api-key-here

# Flask Configuration
FLASK_SECRET_KEY=change-this-to-a-random-string
FLASK_ENV=development
FLASK_DEBUG=True

# Server Configuration
HOST=0.0.0.0
PORT=5000
REDIRECT_URI=http://localhost:5000/oauth2callback
""")
            print("✅ .env file created")
            print("🔧 Please edit .env file with your API keys and credentials")
            return False
    return True

def validate_configuration():
    """Validate that required configuration is present"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']
    api_keys = ['GEMINI_API_KEY', 'HF_API_KEY']
    
    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your-'):
            missing_required.append(var)
    
    missing_api = True
    for key in api_keys:
        value = os.getenv(key)
        if value and not value.startswith('your-'):
            missing_api = False
            break
    
    if missing_required:
        print("❌ Missing required configuration:")
        for var in missing_required:
            print(f"   • {var}")
        return False
    
    if missing_api:
        print("❌ Missing API key configuration:")
        print("   • Set either GEMINI_API_KEY or HF_API_KEY")
        return False
    
    print("✅ Configuration validated")
    return True

def create_templates_directory():
    """Create templates directory and copy HTML file"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("✅ Templates directory created")
    
    # Check if index.html exists in templates
    if not os.path.exists('templates/index.html'):
        print("❌ templates/index.html not found")
        print("   Please ensure the HTML template is in the templates directory")
        return False
    
    return True

def main():
    """Main setup and startup function"""
    print("🤖 RAG-Powered Google Docs Chatbot")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    if not check_virtual_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        print("\n🔧 Configuration needed:")
        print("   1. Edit the .env file with your credentials")
        print("   2. Get Google OAuth credentials from Google Cloud Console")
        print("   3. Get Gemini API key from Google AI Studio")
        print("   4. Run this script again")
        sys.exit(1)
    
    # Validate configuration
    if not validate_configuration():
        print("\n🔧 Please update your .env file with valid credentials")
        sys.exit(1)
    
    # Create templates directory
    if not create_templates_directory():
        sys.exit(1)
    
    print("\n🚀 Starting the application...")
    print("   Open your browser to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the main application
    try:
        from app import app
        app.run(
            debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', 5000))
        )
    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        print("   Make sure app.py is in the current directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()