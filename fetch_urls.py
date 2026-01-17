import requests
from bs4 import BeautifulSoup
import urllib.parse

def fetch_top_n_links(query, num_results):
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', class_='result__a')
        urls = []
        for result in results[:num_results]:
            href = result['href']
            parsed_href = urllib.parse.urlparse(href)
            query_params = urllib.parse.parse_qs(parsed_href.query)
            if 'uddg' in query_params:
                actual_url = query_params['uddg'][0]
                urls.append(actual_url)
            else:
                urls.append(href)

        return urls
    except Exception as e:
        return []