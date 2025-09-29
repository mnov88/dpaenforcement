#!/usr/bin/env python3
"""
Datatilsynet Decision Scraper

This script scrapes decisions from the Norwegian Data Protection Authority (Datatilsynet),
downloads associated PDF documents, extracts text from them, and optionally translates
the content to English using free translation APIs.

Author: [Your name]
Date: 2025
License: MIT
"""

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
from typing import Optional

# PDF text extraction
from pypdf import PdfReader
# Alternative: import pdfplumber

# Translation libraries
from googletrans import Translator
# Alternative: import translators as ts


class DatatilsynetScraper:
    """
    A comprehensive scraper for Datatilsynet (Norwegian DPA) decisions.
    
    Features:
    - Scrapes decision pages from all available years
    - Downloads PDF documents
    - Extracts text from PDFs
    - Translates content using free APIs
    - Exports data to CSV format
    """
    
    def __init__(self, base_url="https://www.datatilsynet.no", download_dir="downloads"):
        self.base_url = base_url
        self.decisions_url = f"{base_url}/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
        # Set up headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize translator
        self.translator = Translator()
        # Default network/backoff settings
        self.default_retries = 3
        self.default_backoff = 1.5
        self.default_page_delay = 1.0

    def fetch(self, url: str, retries: Optional[int] = None, backoff: Optional[float] = None):
        """HTTP GET with simple exponential backoff on transient errors."""
        if retries is None:
            retries = self.default_retries
        if backoff is None:
            backoff = self.default_backoff

        attempt = 0
        delay = 0.5
        while True:
            try:
                response = self.session.get(url, timeout=30)
                # Retry on 429 and 5xx
                if response.status_code in (429,) or 500 <= response.status_code < 600:
                    raise requests.RequestException(f"Transient status {response.status_code}")
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                attempt += 1
                if attempt > retries:
                    raise
                self.logger.warning(f"Transient error on {url}: {e}. Backing off {delay:.1f}s (attempt {attempt}/{retries})")
                time.sleep(delay)
                delay *= backoff
    
    def get_decision_pages_by_year(self, year, max_needed: Optional[int] = None, max_pages: Optional[int] = None, page_delay: Optional[float] = None, retries: Optional[int] = None, backoff: Optional[float] = None):
        """Get decision page links for a specific year with limits/backoff"""
        decision_links = []
        year_url = f"{self.decisions_url}?y={year}"
        
        page = 1
        while True:
            if max_pages is not None and page > max_pages:
                break
            if page == 1:
                url = year_url
            else:
                url = f"{year_url}&p={page}"
            
            try:
                response = self.fetch(url, retries=retries, backoff=backoff)
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find decision links on this page
                links_found = self.extract_decision_links(soup)
                
                if not links_found:
                    break
                    
                decision_links.extend(links_found)
                self.logger.info(f"Found {len(links_found)} decisions on page {page} for {year}")

                # Early cutoff if we already have enough decisions
                if max_needed is not None and len(decision_links) >= max_needed:
                    decision_links = decision_links[:max_needed]
                    break
                
                # Check if there's a next page
                if not self.has_next_page(soup, current_page=page, year=year):
                    break
                    
                page += 1
                time.sleep(self.default_page_delay if page_delay is None else page_delay)  # Be respectful to the server
                
            except requests.RequestException as e:
                self.logger.error(f"Error fetching {url}: {e}")
                break
                
        return decision_links
    
    def extract_decision_links(self, soup):
        """Extract decision page links from listing page"""
        links = []
        
        # Look for links to individual decisions
        decision_pattern = re.compile(r'/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/\d{4}/')
        decision_elements = soup.find_all('a', href=decision_pattern)
        
        for element in decision_elements:
            href = element.get('href')
            if href:
                full_url = urljoin(self.base_url, href)
                title = element.get_text(strip=True)
                if title:  # Only add if we have a title
                    links.append({
                        'url': full_url,
                        'title': title
                    })
                
        return links
    
    def has_next_page(self, soup, current_page: Optional[int] = None, year: Optional[int] = None):
        """Heuristic to check for a next-page link more robustly."""
        # Prefer link elements likely indicating next
        # 1) rel="next"
        link_rel_next = soup.find('a', rel=re.compile(r'next', re.I))
        if link_rel_next:
            return True
        # 2) explicit pagination with p=current_page+1
        if current_page is not None and year is not None:
            next_href = f"?y={year}&p={current_page+1}"
            if soup.find('a', href=re.compile(re.escape(next_href))):
                return True
        # 3) fallback: presence of pagination container with a next-like anchor
        pagination = soup.find(['nav', 'div'], class_=re.compile(r'pagination|pager', re.I))
        if pagination and pagination.find('a', string=re.compile(r'next|neste|›|»', re.I)):
            return True
        return False
    
    def scrape_decision_page(self, decision_url):
        """Scrape metadata from an individual decision page"""
        try:
            response = self.session.get(decision_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                'url': decision_url,
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract title
            title_elem = soup.find('h1')
            metadata['title'] = title_elem.get_text(strip=True) if title_elem else ''
            
            # Extract published date
            date_elem = soup.find(string=re.compile(r'Publisert:'))
            if date_elem:
                date_text = date_elem.parent.get_text(strip=True) if date_elem.parent else str(date_elem)
                metadata['published_date'] = self.extract_date(date_text)
            
            # Extract content summary
            content_area = soup.find('article') or soup.find('div', class_='content')
            if content_area:
                paragraphs = content_area.find_all('p')[:3]  # First 3 paragraphs
                metadata['summary'] = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Find PDF links
            pdf_links = self.find_pdf_links(soup, decision_url)
            metadata['pdf_links'] = pdf_links
            metadata['num_pdfs'] = len(pdf_links)
            
            # Classify decision type
            metadata['decision_type'] = self.classify_decision_type(metadata['title'])
            
            return metadata
            
        except requests.RequestException as e:
            self.logger.error(f"Error scraping {decision_url}: {e}")
            return None
    
    def find_pdf_links(self, soup, base_url):
        """Find all PDF download links on the page"""
        pdf_links = []
        
        # Find PDF links
        pdf_pattern = re.compile(r'\.pdf$', re.IGNORECASE)
        pdf_anchors = soup.find_all('a', href=pdf_pattern)
        
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
    
    def download_pdf(self, pdf_info, download_folder):
        """Download a PDF file"""
        try:
            response = self.session.get(pdf_info['url'])
            response.raise_for_status()
            
            filename = pdf_info['filename']
            filepath = download_folder / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            self.logger.info(f"Downloaded: {filename}")
            return str(filepath)
            
        except requests.RequestException as e:
            self.logger.error(f"Error downloading {pdf_info['url']}: {e}")
            return None
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF using pypdf"""
        try:
            reader = PdfReader(pdf_path)
            text_pages = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text_pages.append({
                    'page': page_num + 1,
                    'text': page_text.strip()
                })
            
            return text_pages
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def translate_text(self, text, target_language='en', source_language='no'):
        """Translate text using free Google Translate"""
        try:
            # Handle long texts by chunking
            if len(text) > 4000:
                chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]
                translated_chunks = []
                
                for chunk in chunks:
                    result = self.translator.translate(chunk, dest=target_language, src=source_language)
                    translated_chunks.append(result.text)
                    time.sleep(0.2)  # Avoid rate limiting
                
                return ' '.join(translated_chunks)
            else:
                result = self.translator.translate(text, dest=target_language, src=source_language)
                return result.text
                
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return text  # Return original if translation fails
    
    def classify_decision_type(self, title):
        """Classify decision type based on Norwegian terms"""
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
        elif 'vedtak' in title_lower:
            return 'Decision'
        else:
            return 'Other'
    
    def extract_date(self, text):
        """Extract Norwegian date format from text"""
        # Pattern: dd.mm.yyyy
        date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
        match = re.search(date_pattern, text)
        
        if match:
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return ''
    
    def get_filename_from_url(self, url):
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or not filename.lower().endswith('.pdf'):
            # Generate filename from URL hash if needed
            filename = f"document_{hash(url) % 10000}.pdf"
            
        return filename
    
    def create_decision_folder(self, title, index):
        """Create a folder for decision files"""
        # Clean title for use as folder name
        safe_title = re.sub(r'[^\w\-_\. ]', '_', title)[:50]
        folder_name = f"{index:03d}_{safe_title}"
        
        decision_folder = self.download_dir / folder_name
        decision_folder.mkdir(exist_ok=True)
        
        return decision_folder
    
    def run_scraper(self, years=None, max_decisions=None, translate=True, target_language='en', map_only=False, max_pages_per_year: Optional[int] = None, page_delay: Optional[float] = None, request_retries: Optional[int] = None, backoff_multiplier: Optional[float] = None):
        """
        Main method to run the complete scraping process
        
        Args:
            years: List of years to scrape (default: [2020-2025])
            max_decisions: Maximum number of decisions to process (for testing)
            translate: Whether to translate content
            target_language: Target language for translation
        """
        if years is None:
            years = [2020, 2021, 2022, 2023, 2024, 2025]
            
        self.logger.info(f"Starting Datatilsynet scraper for years: {years}")
        if map_only:
            self.logger.info("Running in MAP-ONLY mode (no PDF downloads)")
        
        # Apply request defaults from args
        if request_retries is not None:
            self.default_retries = request_retries
        if backoff_multiplier is not None:
            self.default_backoff = backoff_multiplier
        if page_delay is not None:
            self.default_page_delay = page_delay

        # Collect all decision links (respect early cutoff)
        all_decisions = []
        for year in years:
            self.logger.info(f"Collecting decisions from {year}")
            remaining = None
            if max_decisions is not None:
                remaining = max(0, max_decisions - len(all_decisions))
                if remaining == 0:
                    break
            year_decisions = self.get_decision_pages_by_year(
                year,
                max_needed=remaining,
                max_pages=max_pages_per_year,
                page_delay=page_delay,
                retries=request_retries,
                backoff=backoff_multiplier,
            )
            all_decisions.extend(year_decisions)
            time.sleep(self.default_page_delay)
        
        self.logger.info(f"Found {len(all_decisions)} total decisions")
        
        if max_decisions:
            all_decisions = all_decisions[:max_decisions]
            self.logger.info(f"Limited to {len(all_decisions)} decisions for testing")
        
        # Process each decision
        processed_data = []
        
        for i, decision in enumerate(all_decisions, 1):
            self.logger.info(f"Processing {i}/{len(all_decisions)}: {decision['title']}")
            
            # Scrape decision metadata
            metadata = self.scrape_decision_page(decision['url'])
            if not metadata:
                continue
            
            if map_only:
                # In map-only mode, do not download or parse PDFs
                pdf_data = [
                    {
                        'filename': link.get('filename', ''),
                        'url': link.get('url', ''),
                        'link_text': link.get('link_text', '')
                    }
                    for link in metadata.get('pdf_links', [])
                ]
            else:
                # Create folder for this decision
                decision_folder = self.create_decision_folder(metadata['title'], i)
                
                # Process PDFs
                pdf_data = []
                for pdf_info in metadata['pdf_links']:
                    # Download PDF
                    pdf_path = self.download_pdf(pdf_info, decision_folder)
                    if not pdf_path:
                        continue
                    
                    # Extract text
                    text_content = self.extract_pdf_text(pdf_path)
                    
                    # Translate if requested
                    if translate and text_content:
                        for page_data in text_content:
                            if page_data['text']:
                                page_data['translated_text'] = self.translate_text(
                                    page_data['text'], target_language
                                )
                    
                    pdf_data.append({
                        'filename': pdf_info['filename'],
                        'local_path': pdf_path,
                        'text_content': text_content,
                        'url': pdf_info['url']
                    })
            
            # Combine all data
            decision_data = {
                **metadata,
                'pdf_data': pdf_data,
                'processed_at': datetime.now().isoformat()
            }
            
            processed_data.append(decision_data)
            
            # Be respectful to server
            time.sleep(2)
        
        # Save to CSV
        if map_only:
            csv_path = self.save_mapping_csv(processed_data)
        else:
            csv_path = self.save_to_csv(processed_data)
        
        self.logger.info(f"Scraping completed! Processed {len(processed_data)} decisions")
        self.logger.info(f"Data saved to: {csv_path}")
        
        return processed_data
    
    def save_to_csv(self, data):
        """Save processed data to CSV format"""
        csv_data = []
        
        for decision in data:
            base_row = {
                'decision_url': decision['url'],
                'decision_title': decision['title'],
                'published_date': decision.get('published_date', ''),
                'decision_type': decision.get('decision_type', ''),
                'summary': decision.get('summary', ''),
                'num_pdfs': decision.get('num_pdfs', 0),
                'scraped_at': decision.get('scraped_at', ''),
                'processed_at': decision.get('processed_at', '')
            }
            
            if decision['pdf_data']:
                for pdf in decision['pdf_data']:
                    for page_data in pdf['text_content']:
                        row = base_row.copy()
                        row.update({
                            'pdf_filename': pdf['filename'],
                            'pdf_path': pdf['local_path'],
                            'pdf_url': pdf['url'],
                            'page_number': page_data['page'],
                            'original_text': page_data['text'],
                            'translated_text': page_data.get('translated_text', '')
                        })
                        csv_data.append(row)
            else:
                # Add row even if no PDFs
                csv_data.append(base_row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.download_dir / f'datatilsynet_decisions_{timestamp}.csv'
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"CSV saved with {len(csv_data)} rows to {csv_path}")
        return csv_path

    def save_mapping_csv(self, data):
        """Save a lightweight mapping of decisions and PDF links to CSV."""
        rows = []
        for decision in data:
            pdf_links = decision.get('pdf_data', [])
            pdf_urls = [p.get('url', '') for p in pdf_links]
            pdf_filenames = [p.get('filename', '') for p in pdf_links]
            rows.append({
                'decision_url': decision.get('url', ''),
                'decision_title': decision.get('title', ''),
                'published_date': decision.get('published_date', ''),
                'decision_type': decision.get('decision_type', ''),
                'summary': decision.get('summary', ''),
                'num_pdfs': decision.get('num_pdfs', 0),
                'pdf_urls': '|'.join(pdf_urls),
                'pdf_filenames': '|'.join(pdf_filenames),
                'scraped_at': decision.get('scraped_at', ''),
                'processed_at': decision.get('processed_at', '')
            })
        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.download_dir / f'datatilsynet_mapping_{timestamp}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Mapping CSV saved with {len(rows)} rows to {csv_path}")
        return csv_path


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Datatilsynet decisions')
    parser.add_argument('--years', nargs='+', type=int, default=[2023, 2024, 2025],
                        help='Years to scrape (default: 2023-2025)')
    parser.add_argument('--max-decisions', type=int, default=None,
                        help='Maximum number of decisions to process (for testing)')
    parser.add_argument('--no-translate', action='store_true',
                        help='Skip translation step')
    parser.add_argument('--target-lang', default='en',
                        help='Target language for translation (default: en)')
    parser.add_argument('--download-dir', default='datatilsynet_downloads',
                        help='Directory to save downloads (default: datatilsynet_downloads)')
    parser.add_argument('--map-only', action='store_true',
                        help='Only map decisions and PDF URLs; do not download PDFs')
    parser.add_argument('--max-pages-per-year', type=int, default=None,
                        help='Maximum listing pages to crawl per year')
    parser.add_argument('--page-delay', type=float, default=None,
                        help='Delay between listing page requests (seconds)')
    parser.add_argument('--request-retries', type=int, default=None,
                        help='Number of retries for HTTP requests on transient errors')
    parser.add_argument('--backoff-multiplier', type=float, default=None,
                        help='Exponential backoff multiplier for retries (e.g., 1.5)')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = DatatilsynetScraper(download_dir=args.download_dir)
    
    # Run scraper
    data = scraper.run_scraper(
        years=args.years,
        max_decisions=args.max_decisions,
        translate=not args.no_translate,
        target_language=args.target_lang,
        map_only=args.map_only,
        max_pages_per_year=args.max_pages_per_year,
        page_delay=args.page_delay,
        request_retries=args.request_retries,
        backoff_multiplier=args.backoff_multiplier
    )
    
    print(f"\nScraping completed!")
    print(f"Processed {len(data)} decisions")
    print(f"Files saved to: {scraper.download_dir}")


if __name__ == "__main__":
    main()