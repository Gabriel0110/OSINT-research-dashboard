# osint-research-dashboard
Simple dashboard for OSINT research via term query amongst web search and RSS feeds.

## Installation and Usage
1. Clone the repository
2. Install the requirements via `pip install -r requirements.txt`
3. Download the spacy model via `python -m spacy download en_core_web_sm`
4. If you wish to have web search capabilities, setup your Google Custom Search Engine and get the API key and search engine ID. Set the environment variables `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` respectively. They allow 100 search results per day for free, and each query pulls 10.
4. Run the app via `python osint_researcher.py`
5. Open the browser and navigate to `http://127.0.0.1:8050/`

## Notes
1. Email feature currently is Microsoft Outlook client only. Possible more options to be added later, e.g. Gmail.
2. RSS feeds are already included, but feel free to delete the JSON file, or clear it out if you don't want these feeds.
3. Thinking of also adding SerpAPI for additional web search capability (they also have a free amount of usage).
