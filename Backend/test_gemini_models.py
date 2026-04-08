"""Quick test to check available Gemini models"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    
    api_key = os.getenv('ENHANCED_MODEL_KEY')
    if not api_key:
        print("❌ No API key found in .env")
        exit(1)
    
    genai.configure(api_key=api_key)
    
    print("✅ API Key configured successfully")
    print("\n📋 Available models:")
    print("-" * 60)
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display name: {model.display_name}")
            print(f"  Description: {model.description[:80]}...")
            print()
    
except Exception as e:
    print(f"❌ Error: {e}")
