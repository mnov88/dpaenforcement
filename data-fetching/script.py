# Let's create a comprehensive web scraper for Datatilsynet decisions
# I'll demonstrate the complete solution with example code

# First, let's create the basic scraper structure
scraper_code = """
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
import logging
from pathlib import Path

# For PDF text extraction
from pypdf import PdfReader
# Alternative: import pdfplumber

# For translation (using Google Translate API as example)
from googletrans import Translator  # Free client library
# For production, use: from google.cloud import translate_v2 as translate

class DatatilsynetScraper:
    def __init__(self, base_url="https://www.datatilsynet.no", download_dir="downloads"):
        self.base_url = base_url
        self.decisions_url = f"{base_url}/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
        # Set up headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize translator
        self.translator = Translator()
        
    def get_all_decision_pages(self):
        \"\"\"Get all decision pages by year and pagination\"\"\"
        all_decision_links = []
        
        # Get years available (2020-2025 based on our research)
        years = [2020, 2021, 2022, 2023, 2024, 2025]
        
        for year in years:
            self.logger.info(f"Scraping decisions from {year}")
            year_url = f"{self.decisions_url}?y={year}"
            
            page = 1
            while True:
                if page == 1:
                    url = year_url
                else:
                    url = f"{year_url}&p={page}"
                
                try:
                    response = self.session.get(url)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find decision links on this page
                    decision_links = self.extract_decision_links(soup)
                    
                    if not decision_links:
                        break  # No more decisions on this page
                        
                    all_decision_links.extend(decision_links)
                    self.logger.info(f"Found {len(decision_links)} decisions on page {page} for {year}")
                    
                    # Check if there's a next page
                    if not self.has_next_page(soup):
                        break
                        
                    page += 1
                    time.sleep(1)  # Be polite to the server
                    
                except requests.RequestException as e:
                    self.logger.error(f"Error fetching {url}: {e}")
                    break
                    
        return all_decision_links
    
    def extract_decision_links(self, soup):
        \"\"\"Extract individual decision page links from listing page\"\"\"
        links = []
        
        # Look for links to individual decisions
        # Based on the HTML structure we observed
        decision_elements = soup.find_all('a', href=re.compile(r'/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/\d{4}/'))
        
        for element in decision_elements:
            href = element.get('href')
            if href:
                full_url = urljoin(self.base_url, href)
                title = element.get_text(strip=True)
                links.append({
                    'url': full_url,
                    'title': title
                })
                
        return links
    
    def has_next_page(self, soup):
        \"\"\"Check if there's a next page\"\"\"
        # Look for pagination indicators
        pagination = soup.find('div', class_='pagination') or soup.find('nav', class_='pagination')
        if pagination:
            next_link = pagination.find('a', string=re.compile(r'Next|Neste|>', re.I))
            return next_link is not None
        return False
    
    def scrape_decision_page(self, decision_url):
        \"\"\"Scrape an individual decision page for metadata and PDF links\"\"\"
        try:
            response = self.session.get(decision_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            metadata = self.extract_metadata(soup, decision_url)
            
            # Find PDF links
            pdf_links = self.find_pdf_links(soup, decision_url)
            metadata['pdf_links'] = pdf_links
            metadata['num_pdfs'] = len(pdf_links)
            
            return metadata
            
        except requests.RequestException as e:
            self.logger.error(f"Error scraping decision page {decision_url}: {e}")
            return None
    
    def extract_metadata(self, soup, url):
        \"\"\"Extract metadata from decision page\"\"\"
        metadata = {'url': url}
        
        # Title
        title_elem = soup.find('h1')
        metadata['title'] = title_elem.get_text(strip=True) if title_elem else ''
        
        # Date published
        date_elem = soup.find(string=re.compile(r'Publisert:'))
        if date_elem:
            date_parent = date_elem.parent
            date_text = date_parent.get_text(strip=True)
            metadata['published_date'] = self.extract_date(date_text)
        
        # Content/summary
        content_elem = soup.find('div', class_='content') or soup.find('article')
        if content_elem:
            # Get first few paragraphs as summary
            paragraphs = content_elem.find_all('p')[:3]
            metadata['summary'] = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        # Decision type (try to infer from title)
        metadata['decision_type'] = self.classify_decision_type(metadata['title'])
        
        return metadata
    
    def find_pdf_links(self, soup, base_url):
        \"\"\"Find all PDF download links on the page\"\"\"
        pdf_links = []
        
        # Look for direct PDF links
        pdf_anchors = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
        
        for anchor in pdf_anchors:
            href = anchor.get('href')
            if href:
                full_url = urljoin(base_url, href)
                link_text = anchor.get_text(strip=True)
                pdf_links.append({
                    'url': full_url,
                    'link_text': link_text,
                    'filename': self.get_filename_from_url(full_url)
                })
        
        return pdf_links
    
    def download_pdf(self, pdf_info, decision_folder):
        \"\"\"Download a PDF file\"\"\"
        try:
            response = self.session.get(pdf_info['url'])
            response.raise_for_status()
            
            filename = pdf_info['filename']
            filepath = decision_folder / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            self.logger.info(f"Downloaded: {filename}")
            return str(filepath)
            
        except requests.RequestException as e:
            self.logger.error(f"Error downloading {pdf_info['url']}: {e}")
            return None
    
    def extract_pdf_text(self, pdf_path):
        \"\"\"Extract text from PDF\"\"\"
        try:
            reader = PdfReader(pdf_path)
            text_content = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text_content.append({
                    'page': page_num + 1,
                    'text': page_text
                })
            
            return text_content
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def translate_text(self, text, target_language='en'):
        \"\"\"Translate text using free Google Translate\"\"\"
        try:
            if len(text) > 5000:  # Split long texts
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                translated_chunks = []
                for chunk in chunks:
                    result = self.translator.translate(chunk, dest=target_language)
                    translated_chunks.append(result.text)
                    time.sleep(0.1)  # Avoid rate limiting
                return ' '.join(translated_chunks)
            else:
                result = self.translator.translate(text, dest=target_language)
                return result.text
                
        except Exception as e:
            self.logger.error(f"Error translating text: {e}")
            return text  # Return original text if translation fails
    
    def classify_decision_type(self, title):
        \"\"\"Classify decision type based on title\"\"\"
        title_lower = title.lower()
        if 'overtredelsesgebyr' in title_lower:
            return 'Administrative Fine'
        elif 'irettesettelse' in title_lower:
            return 'Reprimand'
        elif 'pålegg' in title_lower:
            return 'Order'
        elif 'forbud' in title_lower:
            return 'Prohibition'
        elif 'tilsyn' in title_lower:
            return 'Supervision'
        else:
            return 'Other'
    
    def extract_date(self, text):
        \"\"\"Extract date from text\"\"\"
        import re
        from datetime import datetime
        
        # Norwegian date pattern: dd.mm.yyyy
        date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
        match = re.search(date_pattern, text)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return ''
    
    def get_filename_from_url(self, url):
        \"\"\"Extract filename from URL\"\"\"
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or not filename.endswith('.pdf'):
            filename = f"document_{hash(url)}.pdf"
        return filename
    
    def run_scraper(self, translate=True, target_language='en'):
        \"\"\"Main method to run the complete scraping process\"\"\"
        self.logger.info("Starting Datatilsynet decision scraper")
        
        # Get all decision pages
        decision_links = self.get_all_decision_pages()
        self.logger.info(f"Found {len(decision_links)} total decisions")
        
        all_data = []
        
        for i, decision_link in enumerate(decision_links[:10], 1):  # Limit to 10 for demo
            self.logger.info(f"Processing decision {i}/{len(decision_links)}: {decision_link['title']}")
            
            # Scrape decision page
            metadata = self.scrape_decision_page(decision_link['url'])
            if not metadata:
                continue
                
            # Create folder for this decision
            safe_title = re.sub(r'[^\w\-_\. ]', '_', metadata['title'])[:50]
            decision_folder = self.download_dir / f"{i:03d}_{safe_title}"
            decision_folder.mkdir(exist_ok=True)
            
            # Download PDFs
            pdf_data = []
            for pdf_info in metadata['pdf_links']:
                pdf_path = self.download_pdf(pdf_info, decision_folder)
                if pdf_path:
                    # Extract text
                    text_content = self.extract_pdf_text(pdf_path)
                    
                    # Translate if requested
                    if translate and text_content:
                        for page_data in text_content:
                            page_data['translated_text'] = self.translate_text(
                                page_data['text'], target_language
                            )
                    
                    pdf_data.append({
                        'filename': pdf_info['filename'],
                        'local_path': pdf_path,
                        'text_content': text_content
                    })
            
            # Add to dataset
            decision_data = {**metadata, 'pdf_data': pdf_data}
            all_data.append(decision_data)
            
            time.sleep(2)  # Be polite to the server
        
        # Save to CSV
        self.save_to_csv(all_data)
        
        return all_data
    
    def save_to_csv(self, data):
        \"\"\"Save scraped data to CSV\"\"\"
        csv_data = []
        
        for decision in data:
            base_row = {
                'url': decision['url'],
                'title': decision['title'],
                'published_date': decision.get('published_date', ''),
                'decision_type': decision.get('decision_type', ''),
                'summary': decision.get('summary', ''),
                'num_pdfs': decision.get('num_pdfs', 0)
            }
            
            if decision['pdf_data']:
                for pdf in decision['pdf_data']:
                    row = base_row.copy()
                    row.update({
                        'pdf_filename': pdf['filename'],
                        'pdf_path': pdf['local_path'],
                        'extracted_text': ' '.join([p['text'] for p in pdf['text_content']]),
                        'translated_text': ' '.join([p.get('translated_text', '') for p in pdf['text_content']])
                    })
                    csv_data.append(row)
            else:
                csv_data.append(base_row)
        
        df = pd.DataFrame(csv_data)
        csv_path = self.download_dir / 'datatilsynet_decisions.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Saved data to {csv_path}")
        return csv_path

# Usage example
if __name__ == "__main__":
    scraper = DatatilsynetScraper()
    
    # Run the scraper with translation
    data = scraper.run_scraper(translate=True, target_language='en')
    
    print(f"Scraped {len(data)} decisions")
"""

print("Complete Datatilsynet scraper code created!")
print("This is a comprehensive solution that includes:")
print("1. Web scraping of decision pages")  
print("2. PDF downloading")
print("3. Text extraction from PDFs")
print("4. Translation using Google Translate API")
print("5. CSV export of all data")