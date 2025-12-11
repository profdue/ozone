import streamlit as st
from supabase import create_client

# Test connection
print("🔍 Testing Supabase connection...")

try:
    # Use your actual credentials
    supabase_url = "https://pqkzgypsllkffqnvdsyn.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBxa3pneXBzbGxrZmZxbnZkc3luIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ5MDI3ODQsImV4cCI6MjA1MDQ3ODc4NH0.46pHbTqJX1cMZqgQ6pIQyPj6h2HxZq8sQZ0YwqRwQ4Y"
    
    client = create_client(supabase_url, supabase_key)
    print("✅ Supabase client created")
    
    # Test insert
    test_data = {
        "home_team": "Test Team",
        "away_team": "Test Opponent",
        "pattern": "TEST CONNECTION",
        "primary_bet": "TEST BET",
        "stake": "1x",
        "confidence": "High",
        "notes": "Testing Supabase connection"
    }
    
    response = client.table("predictions").insert(test_data).execute()
    
    if response.data:
        print(f"✅ Success! Inserted record ID: {response.data[0]['id']}")
        
        # Clean up test record
        client.table("predictions").delete().eq('id', response.data[0]['id']).execute()
        print("✅ Cleaned up test record")
    else:
        print("❌ No data returned")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
