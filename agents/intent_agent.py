import aiohttp
import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager
from datetime import datetime
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

class IntentAgent:
    def __init__(self, db):
        self.db = db
        self.config = ConfigManager()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.user_prompt = None
        self.word2vec_model = None

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    def _train_word2vec(self, content: str):
        sentences = [word_tokenize(sentence.lower()) for sentence in content.split('.')]
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def _get_sentence_vector(self, sentence: str):
        if not self.word2vec_model:
            return None
            
        words = word_tokenize(sentence.lower())
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    async def filter_relevant_content(self, content: str) -> str:
        if not self.user_prompt:
            return content
            
        print(f"\n{Fore.CYAN}Analyzing content for intent: {self.user_prompt}{Style.RESET_ALL}")
        
        # Self-dialogue about user intent
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        intent_analysis = f"""
        Let's analyze the user's intent:
        
        User query: {self.user_prompt}
        Current date: {current_date}
        
        What is the user probably asking about?
        - The user seems to be looking for information about {self.user_prompt}
        - They likely want specific details rather than general information
        - The content should directly answer their query with supporting facts
        - They probably want the most up-to-date information available
        
        How should we filter the content?
        - Look for sections that directly address {self.user_prompt}
        - Include 2-3 surrounding facts for context
        - Prioritize data-driven information over opinions
        - Focus on recent and authoritative sources
        - Give higher priority to content with dates closest to {current_date}
        - If dates are unavailable, prioritize content that appears most current
        - Extract detailed key points with specific citations
        
        What should we exclude?
        - Generic overviews that don't answer the specific query
        - Outdated information (especially older than 1 year)
        - Opinion pieces without factual support
        - Content that only tangentially relates to the query
        - Content with no clear publication date
        """
        
        print(f"{Fore.MAGENTA}Intent analysis:{Style.RESET_ALL}")
        print(intent_analysis)
        
        prompt = f"""
        {intent_analysis}
        
        Now analyze this content:
        {content}
        
        Extract only the sections that directly answer the user's prompt.
        Include 2-3 surrounding facts/context for each relevant section.
        Format the output as markdown with clear section headers.
        
        For each section include:
        - Detailed key points with specific citations
        - Supporting data and statistics
        - Relevant quotes
        - Source references
        - Contextual information
        
        If no relevant content is found, return 'No relevant content found'.
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 5000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    print(f"{Fore.RED}AI filtering error: {error}{Style.RESET_ALL}")
                    return None

    async def process_urls(self, urls: List[str]) -> Dict[str, str]:
        processed_data = {}
        total_urls = len(urls)
        
        with self.db.conn:
            cursor = self.db.conn.cursor()
            
            for i, url in enumerate(urls, 1):
                try:
                    print(f"\n{Fore.BLUE}[{i}/{total_urls}] Processing URL: {url}{Style.RESET_ALL}")
                    
                    cursor.execute('''SELECT content FROM seo_content 
                                   JOIN urls ON seo_content.url_id = urls.id 
                                   WHERE urls.url = ?''', (url,))
                    result = cursor.fetchone()
                    
                    if result and result[0] and not result[0].startswith("Error:"):
                        print(f"{Fore.CYAN}🔍 Filtering content for: '{self.user_prompt}'{Style.RESET_ALL}")
                        print(f"{Fore.WHITE}Content size: {len(result[0])} characters{Style.RESET_ALL}")
                        
                        relevant_content = await self.filter_relevant_content(result[0])
                        
                        if relevant_content and relevant_content != 'No relevant content found':
                            processed_data[url] = relevant_content
                            print(f"{Fore.GREEN}✅ Found {len(relevant_content.splitlines())} relevant sections{Style.RESET_ALL}")
                            print(f"{Fore.WHITE}Relevant content size: {len(relevant_content)} characters{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}⚠️ No content matches the search intent{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}❌ Invalid content - cannot process{Style.RESET_ALL}")
                        
                except Exception as e:
                    print(f"{Fore.RED}❌ Processing failed: {str(e)}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}🎉 URL analysis complete!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Processed {total_urls} URLs, found relevant content in {len(processed_data)} URLs{Style.RESET_ALL}")
        
        # Now process the final summary with epochs
        if processed_data:
            print(f"\n{Fore.CYAN}Starting final summary with 5 epochs...{Style.RESET_ALL}")
            combined_content = "\n\n".join(processed_data.values())
            self._train_word2vec(combined_content)
            
            best_summary = None
            best_score = 0
            
            for epoch in range(1, 6):
                print(f"\n{Fore.BLUE}Epoch {epoch}/5:{Style.RESET_ALL}")
                
                analysis_focus = [
                    "Extract and structure core facts and relationships",
                    "Identify and connect supporting evidence and sources",
                    "Analyze patterns and trends in the information",
                    "Synthesize insights and draw conclusions",
                    "Formulate actionable recommendations"
                ][epoch-1]
                
                analysis_prompt = f"""
                Perform a comprehensive analysis of this content:
                {combined_content}
                
                Analysis Focus: {analysis_focus}
                
                For this iteration, specifically focus on:
                - {analysis_focus}
                - Clear structure and organization
                - Depth of analysis
                - Accuracy of information
                - Relevance to user intent: {self.user_prompt}
                
                Format as:
                ### Executive Summary
                [High-level overview]
                
                ### Key Findings
                [Main insights]
                
                ### Detailed Analysis
                [In-depth examination]
                
                ### Recommendations
                [Actionable suggestions]
                
                ### Sources
                [Citations and references]
                """
                
                payload = {
                    "model": self.config.ai_model,
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "temperature": 0.3 + (epoch * 0.05),
                    "max_tokens": 5000
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            analysis = data['choices'][0]['message']['content']
                            
                            # Use Word2Vec embeddings for semantic similarity
                            if self.word2vec_model:
                                summary_vector = self._get_sentence_vector(analysis)
                                prompt_vector = self._get_sentence_vector(self.user_prompt)
                                if summary_vector is not None and prompt_vector is not None:
                                    semantic_score = np.dot(summary_vector, prompt_vector) / (np.linalg.norm(summary_vector) * np.linalg.norm(prompt_vector))
                                else:
                                    semantic_score = 0
                            else:
                                semantic_score = 0
                            
                            score = self._evaluate_analysis_quality(analysis, epoch)
                            score += semantic_score * 0.2  # Add semantic similarity to score
                            
                            print(f"{Fore.WHITE}Epoch {epoch} quality score: {score:.2f}{Style.RESET_ALL}")
                            print(f"{Fore.WHITE}Semantic similarity: {semantic_score:.2f}{Style.RESET_ALL}")
                            
                            if score > best_score:
                                best_summary = analysis
                                best_score = score
                                print(f"{Fore.GREEN}New best analysis found!{Style.RESET_ALL}")
                            
                            print(f"{Fore.YELLOW}Epoch {epoch} analysis preview:{Style.RESET_ALL}")
                            print(analysis[:200] + "...")
                            
                            combined_content = f"{combined_content}\n\n### Previous Analysis\n{analysis}"
                            
                        else:
                            error = await response.text()
                            print(f"{Fore.RED}Epoch {epoch} failed: {error}{Style.RESET_ALL}")
            
            if best_summary:
                print(f"\n{Fore.GREEN}Final analysis complete! Best score: {best_score:.2f}{Style.RESET_ALL}")
                return best_summary
            else:
                print(f"{Fore.RED}Failed to generate valid analysis{Style.RESET_ALL}")
                return None
        else:
            print(f"{Fore.YELLOW}No relevant content found for summary{Style.RESET_ALL}")
            return None

    def _evaluate_analysis_quality(self, analysis: str, epoch: int) -> float:
        score = 0.0
        
        # Base score for having content
        if analysis:
            score += 0.2
            
        # Structure score (increases weight with epochs)
        structure_components = [
            "### Executive Summary",
            "### Key Findings",
            "### Detailed Analysis",
            "### Recommendations",
            "### Sources"
        ]
        for i, component in enumerate(structure_components):
            if component in analysis:
                score += 0.1 + (0.02 * epoch)  # Increase weight with each epoch
                
        # Content quality (epoch-specific focus)
        if epoch == 1 and "facts" in analysis.lower():
            score += 0.1
        if epoch == 2 and "evidence" in analysis.lower():
            score += 0.1
        if epoch == 3 and "patterns" in analysis.lower():
            score += 0.1
        if epoch == 4 and "insights" in analysis.lower():
            score += 0.1
        if epoch == 5 and "recommendations" in analysis.lower():
            score += 0.1
            
        # Length score (adjusted by epoch)
        score += min(len(analysis) / (2000 + (epoch * 200)), 0.2)
        
        return min(score, 1.0)