import requests
import pandas as pd
import time 

# The official endpoint for twitterapi.io advanced search
url = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# Base query template without a hardcoded static date parameter
base_query = '("Fare hikes" OR "Fuel prices") geocode:-1.286389,36.817223,500km'

# Your authentication header
headers = {
    "x-api-key": "" #USE THE X-API-KEY PROVIDED IN THE 'twitterapi.io/Dashboard' UNDER 'User Information'
}

# --- DATE-CHUNKING ARRAY (From January Last Year to Present) ---
date_chunks = [
    ("2025-01-01", "2025-01-31"),
    ("2025-02-01", "2025-02-28"),
    ("2025-03-01", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"),
    ("2025-11-01", "2025-11-30"),
    ("2025-12-01", "2025-12-31"),
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
    ("2026-04-01", "2026-04-30"),
    ("2026-05-01", "2026-05-31"),
    ("2026-06-01", "2026-06-13") # Tailored to the current date window
]

# --- PAGINATION SETUP ---
MAX_TWEETS = 270  
TWEETS_PER_MONTH = 20  # Quota configuration for monthly constraints
structured_data = []

print(f"Starting extraction... Target: {MAX_TWEETS} maximum cumulative tweets.")

# --- THE TIME-SLICING OUTER LOOP ---
for start_date, end_date in date_chunks:
    # Safely exit if the total historical target threshold is reached
    if len(structured_data) >= MAX_TWEETS:
        break
        
    print(f"\n--- Scraping chunk: {start_date} to {end_date} ---")
    
    # Constructing dynamic query per time parameter
    search_query = f'{base_query} since:{start_date} until:{end_date}'
    
    # Reset tracking elements for the current month execution
    cursor = None
    month_tweets_scraped = 0

    # The Pagination Loop
    while len(structured_data) < MAX_TWEETS:
        
        # Enforce monthly threshold break before querying the endpoint
        if month_tweets_scraped >= TWEETS_PER_MONTH:
            print(f"Monthly quota of {TWEETS_PER_MONTH} tweets met for this segment.")
            break

        querystring = {
            "query": search_query,
            "queryType": "Top"
        }
        
        # If we have a cursor from a previous page, add it to the request
        if cursor:
            querystring["cursor"] = cursor

        try:
            response = requests.get(url, headers=headers, params=querystring)
            
            # If the request is successful
            if response.status_code == 200:
                data = response.json()
                tweets_list = data.get("tweets", [])
                
                # If the page is empty, we've scraped everything available for this query
                if not tweets_list:
                    print("No more tweets found for this query window.")
                    break
                    
                for tweet in tweets_list:
                    # Stop adding data if we hit overall limits or monthly caps
                    if len(structured_data) >= MAX_TWEETS or month_tweets_scraped >= TWEETS_PER_MONTH:
                        break
                        
                    tweet_info = {
                        "Tweet_ID": str(tweet.get("id")), # Kept as string to prevent Excel rounding
                        "Date_Time": tweet.get("createdAt"), 
                        "Username": tweet.get("author", {}).get("userName"),
                        "Tweet_Text": tweet.get("text"),
                        "Likes": tweet.get("likeCount", 0),
                        "Retweets": tweet.get("retweetCount", 0),
                        "Replies": tweet.get("replyCount", 0),
                        "Quotes": tweet.get("quoteCount", 0),
                        "Views": tweet.get("viewCount", 0),
                        "URL": tweet.get("url")
                    }
                    structured_data.append(tweet_info)
                    month_tweets_scraped += 1

                # --- INSERT 1: BULLETPROOF SAVING ---
                # Paste this directly below your 'for' loop, before it searches for the next page
                if structured_data:
                    temp_df = pd.DataFrame(structured_data)
                    temp_df.to_csv("kenyan_transport_sector_structured_SAFE(2).csv", index=False)
                # ------------------------------------
                
                print(f"Successfully scraped {month_tweets_scraped} for this month. Total dataset size: {len(structured_data)} / {MAX_TWEETS} tweets...")
                
                # Find the token for the next page
                has_next = data.get("has_next_page")
                cursor = data.get("next_cursor")
                
                # If there is no next page or if the monthly quota has been populated, break out
                if not has_next or not cursor or month_tweets_scraped >= TWEETS_PER_MONTH:
                    print("Transitioning out of the current operational date window.")
                    break
                    
                # Rate Limit Protection: Sleep for 4 seconds before asking for the next page
                print("Sleeping for 4 seconds to respect API rate limits...")
                time.sleep(4)
                
            # If we hit the rate limit, the server returns a 429 error
            elif response.status_code == 429:
                print("Rate limit hit! Sleeping for 60 seconds to let the server cool down...")
                time.sleep(60)
                
            else:
                print(f"Error {response.status_code}: {response.text}")
                break
                
        # --- INSERT 2: NETWORK SURVIVAL ---
        # Replace your old 'except' block at the very end of the loop with this:
        except Exception as e:
            print(f"Network issue detected! Waiting 30 seconds before retrying... Error: {e}")
            time.sleep(30) # Pauses for 30 seconds
            continue       # Tells Python to try downloading this exact same page again
        # ----------------------------------

# --- FINAL EXPORT ---
if structured_data:
    print("Extraction loop complete. Saving your dataset...")
    df = pd.DataFrame(structured_data)
    
    # Saving to CSV as requested
    df.to_csv("kenyan_transport_sector.csv", index=False)
    #csv file should be named as the folder intended to save the extracted data
    print(f"Success! Data is securely saved to your destination.")
else:
    print("No data was extracted. Please check your query or API key.")