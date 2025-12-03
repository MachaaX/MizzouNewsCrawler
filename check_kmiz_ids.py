#!/usr/bin/env python3
from src.models.database import DatabaseManager
from sqlalchemy import text
import json

# IDs from the list (cleaned up to valid UUIDs)
ids_to_check = [
    '6bf847a0-7c9c-41bd-a8cb-7b47',
    '0f6c9557-bcdf-4f9c-b9c8-f5c4',
    'ec9a8a5f-1cf1-4cd7-af30-9c5c',
    '0fa97464-aed0-4cbe-8079-d600',
    '27e0b6f7-31a1-4cf8-8789-a0b7',
    '6ab6da69-80fd-4238-bf2b-3b1d',
    '0909be12-9176-44bc-b4e4-4cfd',
    '8fd88a5d-3df0-45e7-8ac3-287f',
    '5abd32dc-298c-4720-9e25-d947',
    '8d24a259-ff2e-4305-8d2d-8065'
]

db = DatabaseManager()
with db.get_session() as session:
    # Check as article IDs
    for aid in ids_to_check[:5]:
        result = session.execute(text("""
            SELECT 
                a.id, a.url, a.title, a.author, a.status, a.wire,
                cl.source
            FROM articles a
            JOIN candidate_links cl ON a.candidate_link_id = cl.id
            WHERE a.id = :aid
        """), {'aid': aid}).fetchone()
        
        if result:
            print(f'\n✓ Found as Article ID: {result[0]}')
            print(f'  Source: {result[6]}')
            print(f'  URL: {result[1]}')
            print(f'  Title: {result[2][:80]}')
            print(f'  Author: {result[3]}')
            print(f'  Status: {result[4]}')
            if result[5]:
                try:
                    wire = json.loads(result[5]) if isinstance(result[5], str) else result[5]
                    print(f'  Wire service: {wire.get("service_name", "unknown")}')
                    print(f'  Match type: {wire.get("match_type", "unknown")}')
                    print(f'  Pattern: {wire.get("pattern", "unknown")}')
                except:
                    print(f'  Wire info: {str(result[5])[:100]}')
        else:
            # Try as candidate ID
            result = session.execute(text("""
                SELECT id, url, source, status
                FROM candidate_links
                WHERE id = :cid
            """), {'cid': aid}).fetchone()
            
            if result:
                print(f'\n✓ Found as Candidate ID: {result[0]}')
                print(f'  Source: {result[2]}')
                print(f'  URL: {result[1]}')
                print(f'  Status: {result[3]}')
            else:
                print(f'\n✗ Not found: {aid}')
