"""
SAFE Google Sheets Connection Test
This script uses your local secrets.toml file
"""

import os
import sys

def test_connection():
    print("🔍 SAFE Google Sheets Connection Test")
    print("=" * 60)
    
    # Check if running in Streamlit or locally
    try:
        import streamlit as st
        print("✓ Running in Streamlit environment")
        has_streamlit = True
    except:
        print("✓ Running in local Python environment")
        has_streamlit = False
    
    # Check for secrets file
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        print(f"✓ Found secrets file: {secrets_path}")
    else:
        print(f"✗ Missing secrets file: {secrets_path}")
        print("\nCreate this file with your credentials (it's in .gitignore)")
        return False
    
    print("\n📋 Testing each step...")
    
    try:
        # Step 1: Load secrets safely
        if has_streamlit:
            creds_dict = dict(st.secrets["google_sheets"])
            print("1. ✓ Loaded credentials from Streamlit secrets")
        else:
            # Load from local TOML file
            import toml
            secrets = toml.load(secrets_path)
            creds_dict = secrets["google_sheets"]
            print("1. ✓ Loaded credentials from local secrets file")
        
        # Quick validation
        required_keys = ["private_key_id", "client_email", "project_id"]
        for key in required_keys:
            if key not in creds_dict:
                print(f"   ✗ Missing required key: {key}")
                return False
        
        print(f"   • Project: {creds_dict.get('project_id')}")
        print(f"   • Service Account: {creds_dict.get('client_email')}")
        print(f"   • Key ID: {creds_dict.get('private_key_id')[:8]}...")
        
        # Step 2: Test Google Auth
        from google.oauth2.service_account import Credentials
        import gspread
        
        print("2. Testing Google authentication...")
        credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        print("   ✓ Credentials accepted by Google")
        
        # Step 3: Authorize
        client = gspread.authorize(credentials)
        print("   ✓ Authorization successful")
        
        # Step 4: Test sheet access
        print("3. Testing sheet access...")
        sheet_url = "https://docs.google.com/spreadsheets/d/13Zw4TksoH9P1PWv1HuNpapZOOnT_dIm_Pr85qRof4yE"
        
        try:
            spreadsheet = client.open_by_url(sheet_url)
            print(f"   ✓ Sheet found: {spreadsheet.title}")
            
            # Step 5: Test write
            worksheet = spreadsheet.sheet1
            import datetime
            
            test_data = [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "TEST", "DIAG", "SAFE TEST", "TEST BET",
                "1x", "High", "PENDING", "", "", "Connection working"
            ]
            
            worksheet.append_row(test_data)
            print("   ✓ Write test successful!")
            
            # Clean up
            all_data = worksheet.get_all_values()
            if len(all_data) > 1 and "SAFE TEST" in all_data[-1][3]:
                worksheet.delete_rows(len(all_data))
                print("   ✓ Cleaned up test row")
            
            print("\n" + "=" * 60)
            print("✅ SUCCESS! All tests passed.")
            print("Your Google Sheets integration is working correctly.")
            return True
            
        except gspread.exceptions.APIError as e:
            error_msg = str(e)
            if "PERMISSION_DENIED" in error_msg:
                print(f"   ✗ PERMISSION DENIED")
                print(f"\n💡 SOLUTION: Share your Google Sheet with this email:")
                print(f"   {creds_dict.get('client_email')}")
                print(f"\nSheet URL: {sheet_url}")
            elif "SpreadsheetNotFound" in error_msg:
                print(f"   ✗ Spreadsheet not found")
                print(f"   Check the URL is correct")
            else:
                print(f"   ✗ API Error: {error_msg}")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"✗ ERROR: {error_msg}")
        
        if "invalid_grant" in error_msg.lower():
            print("\n💡 The key format might be wrong. Check:")
            print("   1. private_key has \\n characters (not actual newlines)")
            print("   2. The key starts with '-----BEGIN PRIVATE KEY-----'")
            print("   3. You're using the NEW key (not the revoked one)")
        
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        print("\n🔧 Need to fix something? Follow these steps:")
        print("   1. Check your .streamlit/secrets.toml file exists")
        print("   2. Make sure .streamlit/secrets.toml is in .gitignore")
        print("   3. Share Google Sheet with service account email")
        print("   4. Ensure service account is enabled in Google Cloud")
        sys.exit(1)
