"""
SAFE Google Sheets Connection Test - FORCED LOCAL MODE
"""

import os
import sys

def test_connection():
    print("🔍 Google Sheets Connection Test (Local Mode)")
    print("=" * 60)
    
    # ALWAYS use local mode (ignore Streamlit detection)
    secrets_path = ".streamlit/secrets.toml"
    
    if not os.path.exists(secrets_path):
        print(f"✗ File not found: {secrets_path}")
        print(f"\nExpected location: {os.path.abspath('.')}/{secrets_path}")
        print("\nCreate this file with your TOML-formatted credentials.")
        return False
    
    print(f"✓ Found secrets file: {secrets_path}")
    
    print("\n📋 Testing each step...")
    
    try:
        # Step 1: Load from local TOML file
        import toml
        secrets = toml.load(secrets_path)
        
        if "google_sheets" not in secrets:
            print("✗ Key 'google_sheets' not found in secrets.toml")
            print("\nYour file should start with: [google_sheets]")
            return False
        
        creds_dict = secrets["google_sheets"]
        print("1. ✓ Loaded credentials from local file")
        
        # Quick validation
        print(f"   • Project: {creds_dict.get('project_id', 'MISSING')}")
        print(f"   • Service Account: {creds_dict.get('client_email', 'MISSING')}")
        if 'private_key_id' in creds_dict:
            print(f"   • Key ID: {creds_dict.get('private_key_id')[:8]}...")
        else:
            print("   • Key ID: MISSING")
        
        # Step 2: Test Google Auth
        from google.oauth2.service_account import Credentials
        import gspread
        
        print("2. Testing Google authentication...")
        credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        print("   ✓ Credentials accepted")
        
        client = gspread.authorize(credentials)
        print("   ✓ Authorization successful")
        
        # Step 3: Test sheet access
        print("3. Testing sheet access...")
        sheet_url = "https://docs.google.com/spreadsheets/d/13Zw4TksoH9P1PWv1HuNpapZOOnT_dIm_Pr85qRof4yE"
        
        spreadsheet = client.open_by_url(sheet_url)
        print(f"   ✓ Sheet found: {spreadsheet.title}")
        
        # Step 4: Test write
        worksheet = spreadsheet.sheet1
        import datetime
        
        test_data = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TEST", "DIAG", "LOCAL TEST", "TEST", "1x", "High", "PENDING", "", "", "Connection test"
        ]
        
        worksheet.append_row(test_data)
        print("   ✓ Write test successful!")
        
        # Clean up
        all_data = worksheet.get_all_values()
        if len(all_data) > 1 and "LOCAL TEST" in all_data[-1][3]:
            worksheet.delete_rows(len(all_data))
            print("   ✓ Cleaned up test row")
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Connection is working.")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ ERROR: {error_msg}")
        
        # Specific error handling
        if "invalid_grant" in error_msg.lower():
            print("\n💡 JWT Signature Error - Check:")
            print("   1. private_key has \\n characters in TOML file")
            print("   2. You're using the NEW key (not revoked)")
        elif "PERMISSION_DENIED" in error_msg:
            print(f"\n💡 PERMISSION DENIED")
            print(f"Share sheet with: {creds_dict.get('client_email', 'UNKNOWN')}")
        elif "toml.decoder.TomlDecodeError" in error_msg:
            print("\n💡 TOML Format Error")
            print("Your secrets.toml has wrong syntax")
        elif "module" in error_msg and "toml" in error_msg:
            print("\n💡 Missing 'toml' library")
            print("Run: pip install toml")
        
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        print("\n🔧 Check these common issues:")
        print("   1. Is .streamlit/secrets.toml in the right folder?")
        print("   2. Does it have [google_sheets] at the top?")
        print("   3. Is the private_key formatted correctly?")
        print("   4. Is the service account enabled in Google Cloud?")
        sys.exit(1)
