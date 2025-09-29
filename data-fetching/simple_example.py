#!/usr/bin/env python3
"""
Simple Datatilsynet Scraper Example

A minimal example showing how to scrape Datatilsynet decisions,
download PDFs, and extract/translate text.

This is a simplified version for educational purposes and quick testing.
For production use, see datatilsynet_scraper.py for the full implementation.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time
from urllib.parse import urljoin
from pypdf import PdfReader
from googletrans import Translator


def simple_datatilsynet_scraper(max_decisions=5, translate=True):
    """
    Simple scraper function that demonstrates the basic workflow.
    
    Args:
        max_decisions (int): Maximum number of decisions to process
        translate (bool): Whether to translate text to English
    
    Returns:
        list: List of dictionaries containing scraped data
    """
    print("🚀 Starting simple Datatilsynet scraper...")
    
    # Configuration
    base_url = "https://www.datatilsynet.no"
    decisions_url = f"{base_url}/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/"
    download_dir = "simple_downloads"
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # Initialize session and translator
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    if translate:
        translator = Translator()
    
    results = []
    
    try:
        # Step 1: Get the main decisions page
        print("📄 Fetching decision list...")
        response = session.get(decisions_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Step 2: Find decision links
        decision_links = []
        link_pattern = re.compile(r'/regelverk-og-verktoy/lover-og-regler/avgjorelser-fra-datatilsynet/\d{4}/')
        
        for link in soup.find_all('a', href=link_pattern):
            href = link.get('href')
            title = link.get_text(strip=True)
            if href and title:
                full_url = urljoin(base_url, href)
                decision_links.append({'url': full_url, 'title': title})
        
        print(f"✅ Found {len(decision_links)} decisions")
        
        # Limit for demonstration
        decision_links = decision_links[:max_decisions]
        print(f"📋 Processing {len(decision_links)} decisions...")
        
        # Step 3: Process each decision
        for i, decision in enumerate(decision_links, 1):
            print(f"\n🔄 Processing {i}/{len(decision_links)}: {decision['title'][:60]}...")
            
            try:
                # Get the decision page
                response = session.get(decision['url'])
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract basic metadata
                title_elem = soup.find('h1')
                full_title = title_elem.get_text(strip=True) if title_elem else decision['title']
                
                # Find PDFs on this page
                pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
                print(f"   📎 Found {len(pdf_links)} PDF(s)")
                
                # Process each PDF
                for j, pdf_link in enumerate(pdf_links, 1):
                    pdf_url = urljoin(decision['url'], pdf_link.get('href'))
                    pdf_filename = f"decision_{i}_{j}.pdf"
                    pdf_path = os.path.join(download_dir, pdf_filename)
                    
                    try:
                        # Download PDF
                        print(f"   ⬇️  Downloading PDF {j}...")
                        pdf_response = session.get(pdf_url)
                        pdf_response.raise_for_status()
                        
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_response.content)
                        
                        # Extract text from PDF
                        print(f"   📖 Extracting text...")
                        reader = PdfReader(pdf_path)
                        text_content = ""
                        
                        for page in reader.pages:
                            page_text = page.extract_text()
                            text_content += page_text + "\\n"
                        
                        # Translate if requested
                        translated_text = ""
                        if translate and text_content.strip():
                            print(f"   🌐 Translating...")
                            try:
                                # Limit text length for free API
                                text_to_translate = text_content[:2000]  # First 2000 chars
                                result = translator.translate(text_to_translate, dest='en', src='no')
                                translated_text = result.text
                            except Exception as e:
                                print(f"   ⚠️  Translation failed: {e}")
                                translated_text = "Translation failed"
                        
                        # Store the results
                        results.append({
                            'decision_title': full_title,
                            'decision_url': decision['url'],
                            'pdf_filename': pdf_filename,
                            'pdf_url': pdf_url,
                            'pdf_pages': len(reader.pages),
                            'original_text_preview': text_content[:500],  # First 500 chars
                            'translated_text': translated_text,
                            'file_size_kb': round(len(pdf_response.content) / 1024, 2)
                        })
                        
                        print(f"   ✅ Successfully processed PDF {j}")
                        
                    except Exception as e:
                        print(f"   ❌ Error processing PDF {j}: {e}")
                        continue
                
                # Be respectful to the server
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error processing decision {i}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return results
    
    # Step 4: Save results to CSV
    if results:
        df = pd.DataFrame(results)
        csv_filename = f"{download_dir}/datatilsynet_simple_results.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n✅ Saved {len(results)} results to {csv_filename}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   • Total decisions processed: {len(set([r['decision_title'] for r in results]))}")
        print(f"   • Total PDFs downloaded: {len(results)}")
        print(f"   • Total pages extracted: {sum([r['pdf_pages'] for r in results])}")
        print(f"   • Files saved in: {download_dir}/")
        
        if translate:
            translated_count = len([r for r in results if r['translated_text'] and r['translated_text'] != "Translation failed"])
            print(f"   • Successfully translated: {translated_count} PDFs")
    
    else:
        print("❌ No results collected")
    
    return results


def main():
    """Main function for command line usage"""
    print("🇳🇴 Simple Datatilsynet Decision Scraper")
    print("=" * 50)
    
    # Configuration - modify these as needed
    MAX_DECISIONS = 3  # Start small for testing
    ENABLE_TRANSLATION = True  # Set to False to skip translation
    
    print(f"⚙️  Configuration:")
    print(f"   • Max decisions: {MAX_DECISIONS}")
    print(f"   • Translation: {'Enabled' if ENABLE_TRANSLATION else 'Disabled'}")
    
    # Run the scraper
    results = simple_datatilsynet_scraper(
        max_decisions=MAX_DECISIONS,
        translate=ENABLE_TRANSLATION
    )
    
    if results:
        print(f"\n🎉 Scraping completed successfully!")
        print(f"📁 Check the 'simple_downloads' folder for your files")
        print(f"📊 Open 'datatilsynet_simple_results.csv' to view the data")
    else:
        print(f"\n❌ Scraping failed or no data collected")
    
    print("\n" + "=" * 50)
    print("ℹ️  For production use, see the full 'datatilsynet_scraper.py' script")


if __name__ == "__main__":
    main()