#!/usr/bin/env python3
"""Delete paywall articles from BigQuery."""

from google.cloud import bigquery

# Read article IDs from CSV
article_ids = []
with open('paywall_articles.csv') as f:
    next(f)  # Skip header
    for line in f:
        if line.strip():
            uuid = line.split(',')[0]
            if len(uuid) == 36:  # UUID format
                article_ids.append(uuid)

print(f'Found {len(article_ids)} paywall articles to delete from BigQuery')

# Delete from BigQuery
client = bigquery.Client(project='mizzou-news-projects')
table_id = 'mizzou-news-projects.mizzou_news.articles'

# Delete in batches of 1000
batch_size = 1000
deleted_total = 0

for i in range(0, len(article_ids), batch_size):
    batch = article_ids[i:i+batch_size]
    ids_str = "', '".join(batch)
    
    query = f"""
        DELETE FROM `{table_id}`
        WHERE id IN ('{ids_str}')
    """
    
    job = client.query(query)
    result = job.result()  # Wait for completion
    deleted_total += len(batch)
    print(f'Deleted batch {i//batch_size + 1}: {len(batch)} articles ({deleted_total}/{len(article_ids)} total)')

print(f'\n✅ Successfully deleted {deleted_total} paywall articles from BigQuery')
