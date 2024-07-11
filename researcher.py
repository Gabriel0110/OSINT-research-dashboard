# AIzaSyBtx_z6gCp48K_X5knwi-Umm2QOKc7H4es

"""
<script async src="https://cse.google.com/cse.js?cx=e74c367178c594c5b">
</script>
<div class="gcse-search"></div>
"""

'''
researcher.py
'''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
import feedparser
from datetime import datetime
import pandas as pd
import os
import spacy
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
import networkx as nx
import json
from bs4 import BeautifulSoup
import re
import logging
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, wait, FIRST_COMPLETED
from threading import Lock

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class OSINTResearcher:
    def __init__(self):
        self.rss_feeds = []
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')
        self.nlp = spacy.load("en_core_web_sm")
        self.load_rss_feeds()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def clean_html_and_limit_text(self, html_content, max_length=200):
        # Remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit to max_length characters
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text

    def web_search(self, query):
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
        }
        
        print(f"\nSearching for: {query}")
        response = requests.get(url, params=params)
        results = response.json()

        formatted_results = []
        for item in results.get('items', []):
            formatted_results.append({
                "title": item.get('title', ''),
                "link": item.get('link', ''),
                "snippet": item.get('snippet', ''),
                "source": "Google Custom Search"
            })

        print(f"    - Found {len(formatted_results)} results\n")
        return formatted_results
    
    def load_rss_feeds(self):
        if os.path.exists('rss_feeds.json'):
            with open('rss_feeds.json', 'r') as f:
                self.rss_feeds = json.load(f)

    def save_rss_feeds(self):
        with open('rss_feeds.json', 'w') as f:
            json.dump(self.rss_feeds, f)

    def validate_rss_feed(self, url):
        try:
            feed = feedparser.parse(url)
            if feed.bozo:
                return False, "Invalid RSS feed"
            return True, "Valid RSS feed"
        except Exception as e:
            return False, str(e)

    def add_rss_feed(self, url):
        is_valid, message = self.validate_rss_feed(url)
        if is_valid:
            if url not in self.rss_feeds:
                self.rss_feeds.append(url)
                self.save_rss_feeds()
                return True, "RSS feed added successfully"
            else:
                return False, "RSS feed already exists"
        else:
            return False, "Failed to add RSS feed: " + message

    def remove_rss_feed(self, url):
        if url in self.rss_feeds:
            self.rss_feeds.remove(url)
            self.save_rss_feeds()
            return True, "RSS feed removed successfully"
        return False, "RSS feed not found"

    def get_rss_feeds(self):
        return self.rss_feeds

    def search_rss_feeds(self, query, feed_timeout=20, total_timeout=180):
        results = []
        start_time = time.time()
        total_feeds = len(self.rss_feeds)
        completed_feeds = 0
        completed_lock = Lock()
        
        def search_single_feed(feed_url):
            nonlocal completed_feeds
            try:
                self.logger.info(f"Started searching RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                feed_results = []
                for entry in feed.entries:
                    if query.lower() in entry.title.lower() or query.lower() in entry.get('summary', '').lower():
                        snippet = self.clean_html_and_limit_text(entry.get('summary', ''))
                        feed_results.append({
                            "title": entry.title,
                            "link": entry.link,
                            "snippet": snippet,
                            "source": feed.feed.title,
                            "published": entry.get('published', 'N/A'),
                        })
                with completed_lock:
                    completed_feeds += 1
                    self.logger.info(f"Finished searching RSS feed: {feed_url}. Found {len(feed_results)} results. Progress: {completed_feeds}/{total_feeds}")
                return feed_results
            except Exception as e:
                with completed_lock:
                    completed_feeds += 1
                    self.logger.error(f"Error parsing RSS feed {feed_url}: {e}. Progress: {completed_feeds}/{total_feeds}")
                self.logger.exception("Exception details:")
                return []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(search_single_feed, url): url for url in self.rss_feeds}
            
            while future_to_url:
                done, _ = wait(future_to_url, timeout=min(feed_timeout, max(1, total_timeout - (time.time() - start_time))), return_when=FIRST_COMPLETED)
                
                for future in done:
                    url = future_to_url[future]
                    try:
                        feed_results = future.result(timeout=0)  # Should be instant as it's already done
                        results.extend(feed_results)
                    except TimeoutError:
                        self.logger.error(f"Search for {url} timed out. Progress: {completed_feeds}/{total_feeds}")
                    except Exception as exc:
                        self.logger.error(f'{url} generated an exception: {exc}. Progress: {completed_feeds}/{total_feeds}')
                        self.logger.exception("Exception details:")
                    
                    del future_to_url[future]
                
                if time.time() - start_time > total_timeout:
                    self.logger.warning(f"Total RSS search time exceeded {total_timeout} seconds. Stopping search. Progress: {completed_feeds}/{total_feeds}")
                    break

            # Collect results from any remaining futures after breaking
            for future in future_to_url:
                try:
                    feed_results = future.result(timeout=0)  # Should be instant as it's already done
                    results.extend(feed_results)
                except Exception as exc:
                    self.logger.error(f'Final collection for {future_to_url[future]} generated an exception: {exc}')
                    self.logger.exception("Exception details:")

        self.logger.info(f"Found {len(results)} RSS feed results in {time.time() - start_time:.2f} seconds. Completed {completed_feeds}/{total_feeds} feeds.")
        return results

    def extract_keywords(self, text):
        print("\nExtracting keywords...\n")
        
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return keywords

    def extract_entities(self, text):
        print("\nExtracting entities...\n")
        
        doc = self.nlp(text)
        entities = {
            'PERSON': set(),
            'ORG': set(),
            'GPE': set(),  # Countries, cities, states
            'LOC': set(),  # Non-GPE locations
        }
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].add(ent.text)
        return entities

    def analyze_sentiment(self, text):
        print("\nAnalyzing sentiment...\n")
        
        return TextBlob(text).sentiment.polarity

    def topic_modeling(self, texts, num_topics=5):
        print("\nPerforming topic modeling...\n")
        
        texts = [[word for word in text.lower().split() if word not in stopwords.words('english')]
                 for text in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        return lda_model, corpus, dictionary

    def create_entity_network(self, entities, max_nodes=100, max_edges=1000):
        self.logger.info("Creating entity network...")
        
        G = nx.Graph()
        node_count = 0
        edge_count = 0
        
        # Add nodes
        for entity_type, entity_set in entities.items():
            for entity in entity_set:
                if node_count >= max_nodes:
                    self.logger.warning(f"Reached maximum number of nodes ({max_nodes})")
                    break
                G.add_node(entity, type=entity_type)
                node_count += 1
            if node_count >= max_nodes:
                break
        
        # Add edges
        for entity_type, entities_of_type in entities.items():
            for entity1, entity2 in itertools.combinations(entities_of_type, 2):
                if edge_count >= max_edges:
                    self.logger.warning(f"Reached maximum number of edges ({max_edges})")
                    return G
                if entity1 in G.nodes() and entity2 in G.nodes():
                    G.add_edge(entity1, entity2, type=entity_type)
                    edge_count += 1
        
        self.logger.info(f"Entity network created with {node_count} nodes and {edge_count} edges")
        return G

    def analyze_results(self, results):
        keywords = []
        sources = []
        dates = []
        all_text = ""

        for result in results:
            clean_title = self.clean_html_and_limit_text(result['title'])
            clean_snippet = self.clean_html_and_limit_text(result['snippet'])
            
            keywords.extend(self.extract_keywords(clean_title + ' ' + clean_snippet))
            sources.append(result.get('source', 'Unknown'))
            if 'published' in result:
                try:
                    date = datetime.strptime(result['published'], "%a, %d %b %Y %H:%M:%S %z")
                    dates.append(date)
                except ValueError:
                    pass
            all_text += clean_title + ' ' + clean_snippet + ' '

        keyword_freq = pd.Series(keywords).value_counts().head(10)
        source_dist = pd.Series(sources).value_counts()

        entities = self.extract_entities(all_text)
        sentiment = self.analyze_sentiment(all_text)
        
        texts = [result['title'] + ' ' + result['snippet'] for result in results]
        lda_model, corpus, dictionary = self.topic_modeling(texts)
        
        try:
            entity_network = self.create_entity_network(entities)
        except Exception as e:
            self.logger.error(f"Error creating entity network: {e}")
            self.logger.exception("Exception details:")
            entity_network = nx.Graph()  # Return an empty graph in case of error

        return keyword_freq, source_dist, entities, sentiment, lda_model, corpus, dictionary, entity_network