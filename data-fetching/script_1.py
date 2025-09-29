# Let's create additional code for different translation API options
# and requirements.txt for dependencies

requirements_txt = """# Core web scraping dependencies
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# PDF processing
pypdf>=3.15.0
pdfplumber>=0.9.0
PyMuPDF>=1.23.0  # Alternative: fitz

# Data manipulation
pandas>=2.0.0

# Free translation options
googletrans>=4.0.0  # Free Google Translate client
translators>=5.7.0  # Multiple free translation services

# For production translation APIs (uncomment as needed)
# google-cloud-translate>=3.12.0  # Google Cloud Translation API
# azure-cognitiveservices-language-translator>=3.0.0  # Azure
# boto3>=1.28.0  # For AWS Translate

# Utility libraries
pathlib
urllib3
python-dotenv  # For API keys
"""

# Save requirements.txt content
print("Requirements.txt content:")
print(requirements_txt)
print("\n" + "="*60 + "\n")

# Create translation API examples
translation_examples = """
# Translation API Options for Datatilsynet Scraper

## 1. Free Google Translate (googletrans library)
import googletrans
from googletrans import Translator

class FreeGoogleTranslator:
    def __init__(self):
        self.translator = Translator()
    
    def translate(self, text, target_lang='en', source_lang='no'):
        try:
            # Handle long text by chunking
            if len(text) > 4000:
                chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]
                translated_chunks = []
                for chunk in chunks:
                    result = self.translator.translate(chunk, dest=target_lang, src=source_lang)
                    translated_chunks.append(result.text)
                    time.sleep(0.1)  # Avoid rate limiting
                return ' '.join(translated_chunks)
            else:
                result = self.translator.translate(text, dest=target_lang, src=source_lang)
                return result.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

## 2. Multiple Free Services (translators library)
import translators as ts

class MultiFreeTranslator:
    def __init__(self):
        # Available services: google, bing, baidu, alibaba, yandex, etc.
        self.services = ['google', 'bing', 'yandex']
        self.current_service = 0
    
    def translate(self, text, target_lang='en', source_lang='no'):
        for service in self.services:
            try:
                result = ts.translate_text(text, translator=service, 
                                         from_language=source_lang, 
                                         to_language=target_lang)
                return result
            except Exception as e:
                print(f"Failed with {service}: {e}")
                continue
        return text  # Return original if all fail

## 3. Google Cloud Translation API (Production)
from google.cloud import translate_v2 as translate
import os

class GoogleCloudTranslator:
    def __init__(self, api_key=None):
        if api_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key
        self.client = translate.Client()
    
    def translate(self, text, target_lang='en', source_lang='no'):
        try:
            result = self.client.translate(text, target_language=target_lang, 
                                         source_language=source_lang)
            return result['translatedText']
        except Exception as e:
            print(f"Google Cloud translation error: {e}")
            return text

## 4. Azure Translator (Production)
import requests
import json

class AzureTranslator:
    def __init__(self, api_key, region):
        self.api_key = api_key
        self.region = region
        self.base_url = 'https://api.cognitive.microsofttranslator.com/'
    
    def translate(self, text, target_lang='en', source_lang='no'):
        url = f"{self.base_url}/translate"
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': target_lang
        }
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json'
        }
        body = [{'text': text}]
        
        try:
            response = requests.post(url, params=params, headers=headers, json=body)
            response.raise_for_status()
            result = response.json()
            return result[0]['translations'][0]['text']
        except Exception as e:
            print(f"Azure translation error: {e}")
            return text

## 5. LibreTranslate (Self-hosted/Free)
import requests

class LibreTranslator:
    def __init__(self, api_url="https://libretranslate.com/translate"):
        self.api_url = api_url
    
    def translate(self, text, target_lang='en', source_lang='no'):
        try:
            response = requests.post(self.api_url, data={
                'q': text,
                'source': source_lang,
                'target': target_lang,
                'format': 'text'
            })
            return response.json()['translatedText']
        except Exception as e:
            print(f"LibreTranslate error: {e}")
            return text
"""

print("Translation API Examples:")
print(translation_examples)
print("\n" + "="*60 + "\n")

# Create a simplified version focused on core functionality
simplified_scraper = """
# Simplified Datatilsynet Scraper - Core Functionality

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin
from pypdf import PdfReader
from googletrans import Translator
import time
import re

def scrape_datatilsynet_decisions():
    \"\"\"Simplified scraper for Datatilsynet decisions\"\"\"
    
    base_url = "https://www.datatilsynet.no"
    decisions_url = f"{base_url}/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/"
    
    # Create downloads directory
    os.makedirs("datatilsynet_downloads", exist_ok=True)
    
    # Initialize translator
    translator = Translator()
    
    # Session for requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    decisions_data = []
    
    print("Fetching decision list...")
    response = session.get(decisions_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find decision links (first 10 for demo)
    decision_links = []
    for link in soup.find_all('a', href=re.compile(r'/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/\d{4}/')):
        href = link.get('href')
        title = link.get_text(strip=True)
        if href and title:
            decision_links.append({
                'url': urljoin(base_url, href),
                'title': title
            })
    
    print(f"Found {len(decision_links)} decisions. Processing first 5...")
    
    for i, decision in enumerate(decision_links[:5]):  # Limit for demo
        print(f"Processing {i+1}: {decision['title']}")
        
        try:
            # Get decision page
            response = session.get(decision['url'])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else decision['title']
            
            # Find PDF links
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
            
            for j, pdf_link in enumerate(pdf_links):
                pdf_url = urljoin(decision['url'], pdf_link.get('href'))
                pdf_filename = f"decision_{i+1}_{j+1}.pdf"
                pdf_path = os.path.join("datatilsynet_downloads", pdf_filename)
                
                # Download PDF
                print(f"  Downloading PDF: {pdf_filename}")
                pdf_response = session.get(pdf_url)
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_response.content)
                
                # Extract text from PDF
                try:
                    reader = PdfReader(pdf_path)
                    text_content = ""
                    for page in reader.pages:
                        text_content += page.extract_text() + "\\n"
                    
                    # Translate to English (first 2000 characters to avoid API limits)
                    text_to_translate = text_content[:2000]
                    try:
                        translated = translator.translate(text_to_translate, dest='en', src='no')
                        translated_text = translated.text
                    except:
                        translated_text = "Translation failed"
                    
                    # Store data
                    decisions_data.append({
                        'decision_title': title,
                        'decision_url': decision['url'],
                        'pdf_filename': pdf_filename,
                        'pdf_url': pdf_url,
                        'original_text': text_content[:1000],  # First 1000 chars
                        'translated_text': translated_text,
                        'pdf_pages': len(reader.pages)
                    })
                    
                except Exception as e:
                    print(f"  Error processing PDF {pdf_filename}: {e}")
                
                time.sleep(1)  # Be polite
                
        except Exception as e:
            print(f"Error processing decision: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(decisions_data)
    df.to_csv('datatilsynet_decisions.csv', index=False, encoding='utf-8')
    print(f"Saved {len(decisions_data)} entries to datatilsynet_decisions.csv")
    
    return decisions_data

# Run the scraper
if __name__ == "__main__":
    data = scrape_datatilsynet_decisions()
"""

print("Simplified Scraper Code:")
print(simplified_scraper)