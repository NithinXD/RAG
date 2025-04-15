# Search the web for additional information
def search_web(query, num_results=3):
    try:
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "q": query,
            "num": num_results
        })

        response = requests.post('https://google.serper.dev/search', headers=headers, data=payload)

        if response.status_code == 200:
            results = response.json()
            organic_results = results.get('organic', [])

            formatted_results = []
            for result in organic_results:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                link = result.get('link', '')
                formatted_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n")

            return "\n".join(formatted_results)
        else:
            logger.error(f"Web search failed with status code {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return ""

