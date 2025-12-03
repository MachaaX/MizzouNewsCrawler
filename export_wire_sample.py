#!/usr/bin/env python3
import csv
import sys
from src.models.database import DatabaseManager
from sqlalchemy import text

db = DatabaseManager()
with db.get_session() as session:
    result = session.execute(text("""
        SELECT 
            a.id,
            cl.source as host,
            a.title,
            a.url
        FROM articles a
        JOIN candidate_links cl ON a.candidate_link_id = cl.id
        WHERE a.status = 'wire'
        AND a.extracted_at >= NOW() - INTERVAL '3 days'
        ORDER BY a.extracted_at DESC
        LIMIT 500
    """)).fetchall()
    
    print(f'Found {len(result)} wire articles', file=sys.stderr)
    
    writer = csv.writer(sys.stdout)
    writer.writerow(['uuid', 'host', 'headline', 'url'])
    
    for row in result:
        writer.writerow([str(row[0]), row[1] or '', row[2] or '', row[3] or ''])
