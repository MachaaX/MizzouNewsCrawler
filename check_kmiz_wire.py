#!/usr/bin/env python3
from src.models.database import DatabaseManager
from sqlalchemy import text
import json

db = DatabaseManager()
with db.get_session() as session:
    result = session.execute(text("""
        SELECT 
            a.id, a.url, a.title, a.author, a.status, a.wire, 
            LEFT(a.text, 400) as text_preview
        FROM articles a
        WHERE a.id IN (
            '6bf847a0-7c9c-41bd-a8cb-7b47',
            '0f6c9557-bcdf-4f9c-b9c8-f5c4', 
            'ec9a8a5f-1cf1-4cd7-af30-9c5c',
            '0fa97464-aed0-4cbe-8079-d600',
            '27e0b6f7-31a1-4cf8-8789-a0b7'
        )
    """)).fetchall()
    
    for row in result:
        print(f'\nID: {row[0]}')
        print(f'URL: {row[1]}')
        print(f'Title: {row[2]}')
        print(f'Author: {row[3]}')
        print(f'Status: {row[4]}')
        if row[5]:
            try:
                wire_info = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                print(f'Wire Detection: {json.dumps(wire_info, indent=2)}')
            except:
                print(f'Wire Info (raw): {row[5]}')
        print(f'Text preview: {row[6]}')
        print('=' * 100)
