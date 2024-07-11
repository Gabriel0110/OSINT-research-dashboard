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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from threading import Thread
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class OSINTResearcher:
    def __init__(self):
        self.rss_feeds = []
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("Failed to load spaCy model. Make sure to download the model with 'python -m spacy download en_core_web_sm'")
            print(e)
            exit(1)
            
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

    def search_rss_feeds(self, query):
        results = []
        timeout_per_feed = 30  # seconds
        total_timeout = 180  # seconds

        def search_feed(feed_url):
            try:
                self.logger.info(f"Searching RSS feed: {feed_url}")
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
                return feed_results
            except Exception as e:
                self.logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                self.logger.error(f"Error details: {type(e)}, {e.args}")
                return []

        def run_with_timeout(func, args, timeout):
            result = []
            thread = Thread(target=lambda: result.append(func(*args)))
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                return None
            return result[0]

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(run_with_timeout, search_feed, [url], timeout_per_feed): url for url in self.rss_feeds}
            try:
                for count, future in enumerate(as_completed(future_to_url, timeout=total_timeout), 1):
                    feed_results = future.result()
                    if feed_results is not None:
                        results.extend(feed_results)
                    else:
                        self.logger.warning(f"Timeout or error while searching RSS feed: {future_to_url[future]}")
                    self.logger.info(f"Finished searching feed {future_to_url[future]} ({count}/{len(self.rss_feeds)})")
            except TimeoutError:
                self.logger.error("Total search time exceeded the limit")

        self.logger.info(f"Found {len(results)} RSS feed results\n")
        return results

    def extract_keywords(self, text):
        print("\nExtracting keywords...\n")
        
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return keywords

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {
            'PERSON': set(),
            'ORG': set(),
            'GPE': set(),
            'LOC': set(),
            'DATE': set(),
            'TIME': set(),
            'MONEY': set(),
            'PERCENT': set(),
        }
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].add(ent.text)
        return entities

    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wordcloud.generate(text)
        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img, format='png')
        plt.close()  # Close the figure to free up memory
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

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

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G)

        # Add centrality measures to node attributes
        nx.set_node_attributes(G, degree_centrality, 'degree_centrality')
        nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')
        nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector_centrality')
        
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
        wordcloud_img = self.generate_wordcloud(all_text)

        sentiments = []
        for result in results:
            clean_text = self.clean_html_and_limit_text(result['title'] + ' ' + result['snippet'])
            sentiment = self.analyze_sentiment(clean_text)
            sentiments.append({
                'source': result.get('source', 'Unknown'),
                'published': result.get('published', 'N/A'),
                'sentiment': sentiment
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        sentiment_by_source = sentiment_df.groupby('source')['sentiment'].mean().sort_values(ascending=False)
        
        texts = [result['title'] + ' ' + result['snippet'] for result in results]
        lda_model, corpus, dictionary = self.topic_modeling(texts)
        
        try:
            entity_network = self.create_entity_network(entities)
        except Exception as e:
            self.logger.error(f"Error creating entity network: {e}")
            self.logger.exception("Exception details:")
            entity_network = nx.Graph()  # Return an empty graph in case of error

        return keyword_freq, source_dist, entities, sentiment, lda_model, corpus, dictionary, entity_network, wordcloud_img, sentiment_by_source