# osint-research-dashboard
Simple dashboard for OSINT research via term query amongst web search and RSS feeds.

## Installation and Usage
1. Clone the repository
2. Install the requirements via `pip install -r requirements.txt`
3. Download the spacy model via `python -m spacy download en_core_web_sm`
4. If you wish to have web search capabilities, setup your Google Custom Search Engine and get the API key and search engine ID. Set the environment variables `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` respectively. They allow 100 search results per day for free, and each query pulls 10.
4. Run the app via `python osint_researcher.py`
5. Open the browser and navigate to `http://127.0.0.1:8050/`