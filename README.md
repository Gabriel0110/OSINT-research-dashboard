# osint-research-dashboard
Simple dashboard for OSINT research via term query amongst web search, RSS feeds, and email.

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

<img width="1723" alt="Screenshot 2024-07-20 at 11 17 52 PM" src="https://github.com/user-attachments/assets/1a87f215-bfb0-45ce-933e-c2622c33c9ad">  

<img width="1723" alt="Screenshot 2024-07-20 at 11 18 10 PM" src="https://github.com/user-attachments/assets/d7e78a33-370e-4d2c-8f5e-bea4dbb2edb5">  

<img width="1723" alt="Screenshot 2024-07-20 at 11 18 57 PM" src="https://github.com/user-attachments/assets/bcce9272-c1f2-4b55-820f-b6ccefa24c28">  

<img width="1723" alt="Screenshot 2024-07-20 at 11 19 07 PM" src="https://github.com/user-attachments/assets/9ab21d79-2671-416e-9e2a-c88c76ffcc29">  

<img width="1723" alt="Screenshot 2024-07-20 at 11 19 16 PM" src="https://github.com/user-attachments/assets/b5fd7bbf-730d-41ac-a7db-78e61f653c10">  
  
<img width="1723" alt="Screenshot 2024-07-20 at 11 19 29 PM" src="https://github.com/user-attachments/assets/6ef48368-0a17-41f8-82f4-c5a99882cf66">
