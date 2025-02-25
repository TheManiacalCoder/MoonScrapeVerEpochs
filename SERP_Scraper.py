import requests
import base64
import json
from storage.database_manager import DatabaseManager
from config.manager import ConfigManager
from bs4 import BeautifulSoup
import sqlite3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
import re
from colorama import init, Fore, Style
from agents.analyzer import OpenRouterAnalyzer
import asyncio
import chardet
from agents.content_processor import ContentProcessor
from agents.intent_agent import IntentAgent
import os
import aiohttp
from datetime import datetime

init()  # Initialize Colorama to convert ANSI sequences to Windows equivalents

def show_title_screen():
    print("\n" + "=" * 50)
    print(" " * 20 + "MoonScrape")
    print("=" * 50)
    print(" " * 15 + "A product of OdinWeb3Labs")
    print("=" * 50)
    print("\nInitializing web scraper...")
    print("Establishing database connection...\n")

# Show title screen at startup
show_title_screen()

# Initialize database and config
db = DatabaseManager()
config = ConfigManager()

# Get credentials from config
cred = base64.b64encode(
    f"{config.email}:{config.api_key}".encode()
).decode()

# Set the API endpoint
url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

# Define the payload with the query "who is the current president"
keyword = input("Enter search keyword: ").strip()

payload = [
   {
       "language_code": "en",
       "location_code": 2840,  # United States
       "keyword": keyword
   }
]

# Set headers
headers = {
   "Authorization": f"Basic {cred}",
   "Content-Type": "application/json"
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Add list of domains to exclude
BLACKLISTED_DOMAINS = {
    'reddit.com',
    'youtube.com',
    'vimeo.com',
    'tiktok.com',
    'twitter.com',
    'facebook.com',
    'instagram.com',
    'quora.com',
    'pinterest.com'
}

def is_valid_url(url):
    if not url:
        return False
    return not any(domain in url.lower() for domain in BLACKLISTED_DOMAINS)

# Configure stealth headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def scrape_seo_content(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        if 'charset=' in response.headers.get('content-type', ''):
            encoding = response.headers['content-type'].split('charset=')[-1]
        else:
            detected = chardet.detect(response.content)
            encoding = detected['encoding'] if detected['confidence'] > 0.9 else 'utf-8'
            
        try:
            content = response.content.decode(encoding, errors='strict')
        except UnicodeDecodeError:
            content = response.content.decode(encoding, errors='replace')
            
        error_phrases = [
            'technical difficulties',
            'please try again',
            'forbidden',
            'access denied',
            'unavailable',
            'error occurred',
            'we\'re sorry',
            'temporarily unavailable'
        ]
        
        if any(phrase.lower() in content.lower() for phrase in error_phrases):
            return None
            
        soup = BeautifulSoup(content, 'html.parser')
        
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside', 
                   'form', 'iframe', 'button', 'noscript', 'meta', 'link']:
            for element in soup(tag):
                element.decompose()

        first_heading = None
        for heading_level in range(1, 7):
            first_heading = soup.find(f'h{heading_level}')
            if first_heading:
                break

        if not first_heading:
            first_heading = soup.find(['article', 'main']) or soup.body

        if not first_heading:
            return None

        main_content = []
        current_element = first_heading
        while current_element:
            main_content.append(current_element)
            current_element = current_element.find_next()

        footer = soup.find('footer')
        if footer:
            for element in main_content:
                if element == footer:
                    main_content = main_content[:main_content.index(element)]
                    break

        clean_md = []
        current_list_type = None
        
        for element in main_content:
            try:
                text = element.get_text(' ', strip=True)
                if not text:
                    continue

                if element.name.startswith('h'):
                    level = element.name[1]
                    clean_md.append(f"\n{'#' * int(level)} {text}\n\n")
                    current_list_type = None
                
                elif element.name == 'p':
                    clean_md.append(f"{text}\n\n")
                    current_list_type = None
                
                elif element.name == 'li':
                    list_type = element.find_previous(['ul', 'ol'])
                    if list_type and list_type.name == 'ul':
                        clean_md.append(f"- {text}\n")
                    else:
                        pos = len([li for li in list_type.find_all('li')]) if list_type else 1
                        clean_md.append(f"{pos}. {text}\n")
                    current_list_type = list_type.name if list_type else None
                
                elif element.name == 'blockquote':
                    clean_md.append(f"> {text}\n\n")
                    current_list_type = None

            except Exception as e:
                continue

        formatted = '\n'.join(clean_md)
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        return formatted.strip()
        
    except Exception as e:
        print(f"{Fore.RED}Error scraping {url}: {str(e)}{Style.RESET_ALL}")
        return None

def process_results(items):
    return [item['url'] for item in items if 'url' in item and is_valid_url(item['url'])]

async def run_analysis(collected_urls):
    processor = ContentProcessor(db)
    processed_data = await processor.process_urls(collected_urls)
    
    if processed_data:
        analyzer = OpenRouterAnalyzer(db)
        report = await analyzer.analyze_urls(list(processed_data.keys()))
        if report:
            await analyzer.save_report(report)
    else:
        print(f"{Fore.RED}No valid content to analyze{Style.RESET_ALL}")

def show_progress(step, total_steps, message):
    progress = (step / total_steps) * 100
    print(f"{Fore.CYAN}[{progress:.0f}%] {message}{Style.RESET_ALL}")

async def main():
    try:
        show_progress(0, 4, "Starting search...")
        
        response_data = response.json()
        if not response_data or 'tasks' not in response_data:
            raise ValueError("Invalid API response format")
        
        show_progress(1, 4, "Processing results...")
        
        results = response_data['tasks'][0]['result'][0]
        if not results or 'items' not in results:
            raise ValueError("No search results found")
        
        valid_urls = process_results(results['items'])
        
        show_progress(2, 4, "Checking for additional pages...")
        
        page = 1
        while len(valid_urls) < 10 and page < results.get('metrics', {}).get('pagination', {}).get('total', 1):
            page += 1
            next_payload = [{
                "language_code": "en",
                "location_code": 2840,
                "keyword": keyword,
                "page": page
            }]
            next_response = requests.post(url, headers=headers, json=next_payload)
            next_data = next_response.json()
            if next_data and 'tasks' in next_data:
                valid_urls += process_results(next_data['tasks'][0]['result'][0]['items'])
        
        show_progress(3, 4, "Processing URLs...")
        
        collected_urls = []
        for index, url in enumerate(valid_urls[:10], 1):
            try:
                if not url:
                    continue
                    
                url_id = db.save_url(url)
                
                if url_id is None:
                    with db.conn:
                        url_id = db.conn.execute('SELECT id FROM urls WHERE url = ?', (url,)).fetchone()[0]
                
                content = scrape_seo_content(url)
                if content:
                    db.save_seo_content(url_id, content)
                    print(f"Collected URL {index}: {url}")
                    collected_urls.append(url)
                
            except Exception as e:
                print(f"{Fore.RED}Error processing {url}: {str(e)}{Style.RESET_ALL}")
        
        show_progress(4, 4, "Search complete!")
        
        print(f"\n{Fore.CYAN}Starting intent-based filtering...{Style.RESET_ALL}")
        intent_agent = IntentAgent(db)
        intent_agent.set_prompt(keyword)
        final_summary = await intent_agent.process_urls(collected_urls)
        
        if final_summary:
            print(f"\n{Fore.MAGENTA}Preparing final summary for analysis...{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Summary length: {len(final_summary)} characters{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Executive Summary Preview:{Style.RESET_ALL}")
            
            exec_start = final_summary.find("### Executive Summary")
            if exec_start != -1:
                exec_end = final_summary.find("###", exec_start + 1)
                if exec_end == -1:
                    exec_end = len(final_summary)
                
                exec_summary = final_summary[exec_start:exec_end]
                
                print(exec_summary[:600] + "...")
                
                if len(exec_summary) > 600:
                    print("\n...")
                    print(exec_summary[-200:])
            else:
                print(final_summary[:600] + "...")
        
        print("\nStarting comprehensive SEO analysis...")
        analyzer = OpenRouterAnalyzer(db)
        analyzer.set_prompt(keyword)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if final_summary:
            final_summary = f"""
            {final_summary}
            
            Important:
            - Verify all information is current as of {current_date}
            - Reject outdated information
            - Prioritize recent sources
            """
        
        summary_data = {"final_summary": final_summary}
        report = await analyzer.analyze_urls(summary_data)
        
        if report:
            await analyzer.save_report(report)
            print(f"{Fore.GREEN}Comprehensive SEO analysis complete!{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Final Report:{Style.RESET_ALL}")
            print(report)
            
            # Display collected URLs
            print(f"\n{Fore.MAGENTA}Source URLs:{Style.RESET_ALL}")
            if collected_urls:
                for i, url in enumerate(collected_urls, 1):
                    print(f"{i}. {url}")
            else:
                print("No source URLs found in the report")
        else:
            print(f"{Fore.RED}Failed to generate analysis report{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error in main workflow: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())
