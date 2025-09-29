# Datatilsynet Decision Scraper

A comprehensive Python script to scrape decisions from the Norwegian Data Protection Authority (Datatilsynet), download associated PDF documents, extract text content, and optionally translate the content to English using free translation APIs.

## Features

- **Complete Web Scraping**: Automatically discovers and scrapes all decision pages from Datatilsynet's website
- **PDF Download**: Downloads all PDF documents associated with each decision
- **Text Extraction**: Extracts text content from PDFs using multiple libraries (pypdf, pdfplumber, PyMuPDF)  
- **Free Translation**: Translates Norwegian content to English using free translation APIs
- **CSV Export**: Exports all data to CSV format for easy analysis
- **Respectful Scraping**: Implements delays and proper headers to be respectful to the server
- **Error Handling**: Robust error handling and logging throughout the process
- **Flexible Configuration**: Command-line options for years, limits, and translation settings

## Installation

1. **Clone or download the files**:
   - `datatilsynet_scraper.py` - Main scraper script
   - `requirements.txt` - Python dependencies
   - `README.md` - This documentation

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Alternative installation**:
   ```bash
   # Core dependencies
   pip install requests beautifulsoup4 pandas pypdf googletrans==4.0.0
   
   # Optional PDF libraries
   pip install pdfplumber PyMuPDF
   
   # Alternative translation libraries  
   pip install translators
   ```

## Basic Usage

### Simple Run (Recommended for testing)
```bash
# Scrape last 3 years with translation, limit to 10 decisions for testing
python datatilsynet_scraper.py --years 2023 2024 2025 --max-decisions 10
```

### Full Run (Production)
```bash
# Scrape all available years with full translation
python datatilsynet_scraper.py --years 2020 2021 2022 2023 2024 2025
```

### Custom Options
```bash
# Scrape specific years, no translation, custom directory
python datatilsynet_scraper.py --years 2024 2025 --no-translate --download-dir my_downloads

# Translate to German instead of English
python datatilsynet_scraper.py --target-lang de --max-decisions 5
```

## Command Line Options

- `--years YEARS [YEARS ...]`: Years to scrape (default: 2023 2024 2025)
- `--max-decisions N`: Maximum number of decisions to process (useful for testing)
- `--no-translate`: Skip translation step (faster processing)
- `--target-lang LANG`: Target language code for translation (default: 'en')
- `--download-dir DIR`: Directory to save downloads (default: 'datatilsynet_downloads')

## Output Structure

The scraper creates the following structure:

```
datatilsynet_downloads/
├── 001_Decision_Title_1/
│   ├── document1.pdf
│   └── document2.pdf
├── 002_Decision_Title_2/
│   └── decision.pdf
├── datatilsynet_decisions_20250929_143022.csv
└── scraper.log
```

### CSV Output Format

The generated CSV contains these columns:
- `decision_url`: URL of the decision page
- `decision_title`: Title of the decision
- `published_date`: When the decision was published (YYYY-MM-DD format)
- `decision_type`: Classified type (Administrative Fine, Reprimand, Order, etc.)
- `summary`: Brief summary from the decision page
- `num_pdfs`: Number of PDF files found
- `pdf_filename`: Name of the PDF file
- `pdf_path`: Local path to downloaded PDF
- `pdf_url`: Original URL of the PDF
- `page_number`: Page number within the PDF
- `original_text`: Extracted Norwegian text
- `translated_text`: English translation (if enabled)
- `scraped_at`: When the decision page was scraped
- `processed_at`: When the processing was completed

## Translation Options

### Free Translation APIs

The script supports multiple free translation options:

1. **Google Translate (default)** - Uses `googletrans` library
2. **Multiple services** - Uses `translators` library (Google, Bing, Yandex)  
3. **LibreTranslate** - Self-hosted or public instance

### Production Translation APIs

For higher volume or production use, consider these paid options:

1. **Google Cloud Translation API**
2. **Microsoft Azure Translator**
3. **Amazon Translate**

See the script comments for implementation details.

## Important Considerations

### Legal and Ethical

- ✅ **Legally compliant**: Datatilsynet decisions are public documents
- ✅ **Open source**: Datatilsynet explicitly publishes decisions as open data
- ✅ **Respectful scraping**: Script includes delays and proper user agent headers
- ✅ **Research purpose**: Intended for academic and legal research

### Rate Limiting and Courtesy

The script includes several measures to be respectful:
- 1-2 second delays between requests
- Proper User-Agent headers
- Error handling for server issues
- Logging of all activities

### Translation Limitations

- **Free APIs**: May have daily limits (typically 100K-500K characters)
- **Quality**: Free translation may not be perfect for legal terminology
- **Rate limits**: Script includes delays to avoid hitting API limits

## Troubleshooting

### Common Issues

1. **Translation errors**: Often due to API rate limits
   - Solution: Use `--no-translate` flag or reduce `--max-decisions`

2. **PDF extraction errors**: Some PDFs may be scanned images
   - Script will log errors and continue with other documents

3. **Network timeouts**: 
   - Script automatically retries and logs errors
   - Check internet connection and Datatilsynet website status

### Error Logs

Check `scraper.log` for detailed error information:
```bash
tail -f scraper.log  # Monitor in real-time
grep ERROR scraper.log  # Find all errors
```

## Example Results

After running the scraper, you'll have:

- **~100-400 decisions** (depending on years selected)
- **~200-800 PDF documents** downloaded locally
- **Full text content** extracted from all PDFs
- **English translations** of all text (if enabled)
- **Structured CSV** ready for analysis in Excel, R, Python, etc.

## Data Analysis Ideas

With the scraped data, you could analyze:
- Trends in fine amounts over time
- Most common GDPR violations
- Processing times for different case types
- Geographic distribution of cases
- Comparative analysis with other EU DPAs

## Advanced Usage

### Custom PDF Processing

You can modify the script to use different PDF libraries:

```python
# Instead of pypdf, use pdfplumber for better table extraction
import pdfplumber

def extract_pdf_text_advanced(self, pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\\n"
            # Also extract tables
            tables = page.extract_tables()
            for table in tables:
                # Process table data
                pass
        return text
```

### Custom Translation Services

You can easily swap translation providers:

```python
# Use different translation service
import translators as ts

def translate_text_multi(self, text, target_lang='en'):
    try:
        return ts.translate_text(text, translator='bing', to_language=target_lang)
    except:
        return ts.translate_text(text, translator='yandex', to_language=target_lang)
```

## Contributing

This script can be extended to:
- Add support for other Norwegian legal databases
- Implement OCR for scanned PDFs
- Add more sophisticated text processing
- Create automated analysis pipelines
- Build web interface or API

## Disclaimer

This tool is for research and educational purposes. Please:
- Use responsibly and respectfully
- Don't overload Datatilsynet's servers
- Respect any robots.txt or terms of service
- Consider the computational and network resources involved

## Support

For questions about:
- **GDPR/Legal research**: Contact Norwegian legal research institutions
- **Python/Technical issues**: Check the error logs and documentation
- **Datatilsynet website changes**: May require script updates

## License

MIT License - Feel free to modify and extend for your research needs.