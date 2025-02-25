import aiohttp
import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager
from datetime import datetime
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from pathlib import Path

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
        self.model_path = Path("models/word2vec")
        self.model_path.mkdir(parents=True, exist_ok=True)

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    def _train_word2vec(self, content: str):
        model_file = self.model_path / "word2vec.model"
        
        if model_file.exists():
            self.word2vec_model = Word2Vec.load(str(model_file))
            return
            
        sentences = [word_tokenize(sentence.lower()) for sentence in content.split('.') if sentence.strip()]
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
        self.word2vec_model.save(str(model_file))

    def _get_sentence_vector(self, sentence: str):
        if not self.word2vec_model:
            return None
            
        words = word_tokenize(sentence.lower())
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        vec1 = self._get_sentence_vector(text1)
        vec2 = self._get_sentence_vector(text2)
        
        if vec1 is None or vec2 is None:
            return 0.0
            
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return max(0.0, min(1.0, cosine_sim))

    async def filter_relevant_content(self, content: str) -> str:
        if not self.user_prompt:
            return content
            
        print(f"\n{Fore.CYAN}Analyzing content for intent: {self.user_prompt}{Style.RESET_ALL}")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
        timestamp_context = f"""
        Current Date: {current_date}
        Current Year: {current_year}
        
        Important: 
        - Verify all information is current as of {current_date}
        - Reject outdated information
        - Prioritize recent sources
        - Flag content older than 1 year
        - Ensure temporal accuracy
        """
        
        intent_analysis = f"""
        {timestamp_context}
        
        Analyze this query deeply: "{self.user_prompt}"
        
        First, verify temporal context:
        1. Does the query require current information? [Yes/No]
        2. Does the content contain timestamps? [Yes/No]
        3. Is the content outdated? [Yes/No]
        
        Then determine temporal accuracy:
        - If content is outdated: [Mark as invalid]
        - If content is recent: [Mark as valid]
        - If content lacks timestamps: [Verify with external sources]
        """
        
        prompt = f"""
        {intent_analysis}
        
        Now analyze this content:
        {content}
        
        Apply temporal filtering based on the query analysis.
        Exclude outdated content unless explicitly requested.
        Ensure all information is current as of {current_date}.
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
                    print(f"{Fore.RED}Error filtering content: {error}{Style.RESET_ALL}")
                    return None

    async def _spell_check_query(self, query: str) -> str:
        prompt = f"""
        Review this search query for spelling/grammar issues:
        "{query}"
        
        Return ONLY: 
        - The corrected query if errors found
        - The original query if no errors
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip('"')
                return query

    async def process_urls(self, urls: List[str]) -> Dict[str, str]:
        processed_data = {}
        total_urls = len(urls)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        with self.db.conn:
            cursor = self.db.conn.cursor()
            
            for i, url in enumerate(urls, 1):
                try:
                    print(f"\nProcessing URL {i}/{total_urls}: {url}")
                    
                    cursor.execute('''SELECT content FROM seo_content 
                                   JOIN urls ON seo_content.url_id = urls.id 
                                   WHERE urls.url = ?''', (url,))
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        content = result[0]
                        
                        if "current" in self.user_prompt.lower() or "today" in self.user_prompt.lower():
                            content = f"""
                            Current Date: {current_date}
                            
                            {content}
                            
                            Important:
                            - Verify all information is current as of {current_date}
                            - Reject outdated information
                            - Prioritize recent sources
                            """
                            
                        processed_data[url] = content
                except Exception as e:
                    print(f"Processing failed: {str(e)}")
        
        print(f"\nURL analysis complete! Processed {total_urls} URLs, found relevant content in {len(processed_data)} URLs")
        
        if processed_data:
            print(f"\nStarting final summary with 5 epochs...")
            combined_content = "\n\n".join(processed_data.values())
            self._train_word2vec(combined_content)
            
            best_summary = None
            best_score = 0
            previous_analysis = None
            
            for epoch in range(1, 6):
                print(f"\nEpoch {epoch}/5:")
                
                analysis_focus = [
                    "Extract and present core facts and data points",
                    "Identify and present supporting evidence and sources",
                    "Present patterns and trends with specific data",
                    "Synthesize findings with direct evidence",
                    "Formulate recommendations based on presented facts"
                ][epoch-1]
                
                analysis_prompt = f"""
                Analyze this content and present factual information:
                {combined_content}
                
                Focus: {analysis_focus}
                
                Requirements:
                - Present specific facts and data
                - Include source references
                - Use direct quotes where relevant
                - Maintain factual accuracy
                - Build upon previous analysis if available
                - Focus on evidence-based presentation
                
                Previous Analysis:
                {previous_analysis if previous_analysis else "No previous analysis"}
                
                Format:
                - Fact: [specific fact]
                - Source: [source reference]
                - Evidence: [supporting data/quote]
                - Context: [relevant background]
                - Analysis: [your insights]
                
                Example:
                Fact: Joe Biden is the current US President
                Source: White House website
                Evidence: Official inauguration date January 20, 2021
                Context: 46th President, won 2020 election
                Analysis: This confirms the current political leadership
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
                            
                            semantic_score = self._calculate_semantic_similarity(analysis, self.user_prompt)
                            score = self._evaluate_analysis_quality(analysis, epoch)
                            score += semantic_score * 0.2
                            
                            print(f"Epoch {epoch} quality score: {score:.2f}")
                            print(f"Semantic similarity: {semantic_score:.2f}")
                            
                            if score > best_score:
                                best_summary = analysis
                                best_score = score
                                print("New best analysis found!")
                            
                            print(f"Epoch {epoch} analysis preview:")
                            print(analysis[:800] + "...")
                            
                            previous_analysis = analysis
                            combined_content = f"{combined_content}\n\n### Previous Analysis\n{analysis}"
                            
                        else:
                            error = await response.text()
                            print(f"Epoch {epoch} failed: {error}")
            
            if best_summary:
                print(f"\nFinal analysis complete! Best score: {best_score:.2f}")
                return best_summary
            else:
                print("Failed to generate valid analysis")
                return None
        else:
            print("No relevant content found for summary")
            return None

    def _evaluate_analysis_quality(self, analysis: str, epoch: int) -> float:
        score = 0.0
        
        # Base score for having content
        if analysis:
            score += 0.2
            
        # Freshness score (increases weight with epochs)
        current_year = datetime.now().year
        if str(current_year) in analysis:
            score += 0.1 + (0.02 * epoch)
            
        # Timestamp score
        if "as of" in analysis.lower() or "current" in analysis.lower():
            score += 0.1
            
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
                score += 0.1 + (0.02 * epoch)
                
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
        
        # Semantic similarity score (increases weight with epochs)
        if self.word2vec_model:
            semantic_score = self._calculate_semantic_similarity(analysis, self.user_prompt)
            score += semantic_score * (0.1 + (0.02 * epoch))
        
        # Clarity score (increases with epochs)
        if "clearly" in analysis.lower() or "concisely" in analysis.lower():
            score += 0.05 * epoch
            
        # Depth score (increases with epochs)
        depth_indicators = ["detailed", "in-depth", "comprehensive", "thorough"]
        for indicator in depth_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        # Specificity score (increases with epochs)
        if "specific" in analysis.lower() or "precise" in analysis.lower():
            score += 0.05 * epoch
            
        # Evidence score (increases with epochs)
        evidence_indicators = ["data", "statistics", "research", "study", "source"]
        for indicator in evidence_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        # Actionability score (increases with epochs)
        if "actionable" in analysis.lower() or "recommendation" in analysis.lower():
            score += 0.05 * epoch
            
        return min(score, 1.0)