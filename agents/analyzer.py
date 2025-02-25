import os
import json
import aiohttp
import asyncio
from config.manager import ConfigManager
from pathlib import Path
from colorama import Fore, Style
from typing import List
import re
from datetime import datetime

class OpenRouterAnalyzer:
    def __init__(self, db):
        self.config = ConfigManager()
        self.db = db
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.analysis_folder = Path("analysis")
        self.analysis_folder.mkdir(exist_ok=True)
        self.user_prompt = None

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    async def analyze_urls(self, filtered_content: dict):
        try:
            print(f"{Fore.CYAN}Performing comprehensive SEO analysis...{Style.RESET_ALL}")
            
            if "final_summary" not in filtered_content:
                raise ValueError("Expected final summary data")
                
            if not self.user_prompt:
                raise ValueError("User prompt not set")
                
            summary = filtered_content["final_summary"]
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            prompt = f"""
            Current Date: {current_date}
            Analysis Epoch: 1
            
            Directly answer this question: {self.user_prompt}
            Always give 1o key points relative to the query: {self.user_prompt}
            
            Use this content as your source:
            {summary}
            
            Requirements:
            - Verify information is current as of {current_date}
            - Cross-check with official sources
            - Reject outdated information
            - Prioritize primary sources
            - Include timestamp verification
            - Format as a direct response
            
            Verification Process:
            1. Check source timestamps
            2. Verify with official sources
            3. Reject outdated claims
            4. Confirm with current data
            5. Include verification details
            
            Example format:
            The current US President is Donald Trump (verified as of {current_date}). 
            Source: White House website (verified)
            Evidence: Official inauguration date January 20, 2025
            Verification: Confirmed current as of {current_date}
            """
            
            payload = {
                "model": self.config.ai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 3000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error = await response.text()
                        raise Exception(f"OpenRouter API error: {error}")
                        
        except Exception as e:
            print(f"{Fore.RED}Error during analysis: {e}{Style.RESET_ALL}")
            return None

    def _sort_urls_by_relevance(self, content: str) -> List[str]:
        # Extract URLs from content
        urls = re.findall(r'https?://[^\s]+', content)
        
        # Score URLs based on relevance factors
        scored_urls = []
        for url in urls:
            score = 0
            
            # Higher score for main domain mentions
            domain = re.sub(r'https?://(www\.)?', '', url)
            domain = re.sub(r'\/.*', '', domain)
            score += content.lower().count(domain) * 0.1
            
            # Higher score for exact URL mentions
            score += content.lower().count(url) * 0.2
            
            # Higher score for earlier mentions
            position = content.lower().find(url)
            if position != -1:
                score += (1 - (position / len(content))) * 0.3
                
            # Higher score for authoritative domains
            if any(auth in domain for auth in ['.gov', '.edu', '.org']):
                score += 0.2
                
            scored_urls.append((url, score))
        
        # Sort by score descending
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return [url for url, score in scored_urls]

    def _get_content_for_url(self, url):
        # Implement content retrieval from your database or storage
        pass

    async def save_report(self, report):
        report_path = self.analysis_folder / "aggregated_analysis.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"{Fore.GREEN}Aggregated report saved to {report_path}{Style.RESET_ALL}")

async def main(urls):
    analyzer = OpenRouterAnalyzer()
    report = await analyzer.analyze_urls(urls)
    if report:
        await analyzer.save_report(report)

# Example usage
if __name__ == "__main__":
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5"
    ]
    asyncio.run(main(urls)) 