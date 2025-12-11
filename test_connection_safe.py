"""
CORRECT Google Sheets Connection Test
Reads from local .streamlit/secrets.toml file only
"""

import os
import sys
import toml  # Make sure this is installed: pip install toml

def test_connection():
    print("🔍 Google Sheets Connection Test (Local File)")
    print("=" * 60)
    
    secrets_path = ".streamlit/secrets.toml"
    
    # Check if file exists
    if not os.path.exists(secrets_path):
        print(f"❌ ERROR: File not found: {secrets_path}")
        print(f"Expected: {os.path.abspath('.')}/{secrets_path}")
        print("\nMake sure your folder structure is:")
        print("  project-folder/")
        print("  ├── app.py")
        print("  ├── test_connection_safe.py")
        print("  └── .streamlit/")
        print("      └── secrets.toml  <-- This file must exist")
        return False
    
    print(f"✅ Found: {secrets_path}")
    
    try:
        # STEP 1: Load and check the TOML file
        print("\n1. Reading secrets.toml file...")
        secrets = toml.load(secrets_path)
        
        # Check for the google_sheets section
        if "google_sheets" not in secrets:
            print("❌ ERROR: Missing [google_sheets] section in secrets.toml")
            print("\nYour file should start with:")
            print("[google_sheets]")
            print("type = \"service_account\"")
            print("project_id = \"football-prediction-tracker\"")
            print("... etc")
            return False
        
        creds = secrets["google_sheets"]
        print("✅ [google_sheets] section found")
        print(f"   • Project: {creds.get('project_id')}")
        print(f"   • Account: {creds.get('client_email')}")
        
        # STEP 2: Test Google authentication
        print("\n2. Testing Google authentication...")
        from google.oauth2.service_account import Credentials
        import gspread
        
        credentials = Credentials.from_service_account_info(
            creds,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        client = gspread.authorize(credentials)
        print("✅ Google authentication successful")
        
        # STEP 3: Test sheet access
        print("\n3. Testing sheet access...")
        sheet_url = "https://docs.google.com/spreadsheets/d/13Zw4TksoH9P1PWv1HuNpapZOOnT_dIm_Pr85qRof4yE"
        
        spreadsheet = client.open_by_url(sheet_url)
        print(f"✅ Sheet found: {spreadsheet.title}")
        
        # STEP 4: Test write permission
        print("\n4. Testing write permission...")
        worksheet = spreadsheet.sheet1
        import datetime
        
        test_row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TEST", "DIAG", "LOCAL TEST", "TEST BET",
            "1x", "High", "PENDING", "", "", "Diagnostic test"
        ]
        
        worksheet.append_row(test_row)
        print("✅ Write test successful! Row added to sheet.")
        
        # Clean up
        all_data = worksheet.get_all_values()
        if len(all_data) > 1 and "LOCAL TEST" in all_data[-1][3]:
            worksheet.delete_rows(len(all_data))
            print("✅ Cleaned up test row")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Your Google Sheets connection is working perfectly.")
        print("\nNext: Run 'streamlit run app.py' and test the save button.")
        return True
        
    except ImportError as e:
        print(f"❌ Missing Python library: {str(e)}")
        print("\nInstall missing library:")
        print("  pip install toml gspread google-auth")
        return False
        
    except toml.decoder.TomlDecodeError as e:
        print(f"❌ Invalid TOML format: {str(e)}")
        print("\nCheck your secrets.toml file syntax.")
        return False
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: {error_msg}")
        
        # Specific error handling
        if "invalid_grant" in error_msg.lower():
            print("\n🔍 ISSUE: Invalid JWT Signature")
            print("Your private key format is wrong in secrets.toml")
            print("Make sure private_key has \\n characters (not actual newlines)")
            
        elif "PERMISSION_DENIED" in error_msg:
            print("\n🔍 ISSUE: Permission Denied")
            print(f"Share your Google Sheet with this email:")
            print(f"  {creds.get('client_email', 'UNKNOWN')}")
            print(f"\nSheet URL: {sheet_url}")
            
        elif "SpreadsheetNotFound" in error_msg:
            print("\n🔍 ISSUE: Spreadsheet not found")
            print("Check the sheet URL in your code")
            
        elif "Unable to discover service account" in error_msg:
            print("\n🔍 ISSUE: Service account disabled")
            print("Go to Google Cloud Console > IAM > Service Accounts")
            print("Ensure 'ozone-football' is ENABLED")
            
        else:
            print(f"\n🔍 UNKNOWN ERROR: Check your setup")
            
        return False

if __name__ == "__main__":
    # First, check if we have the required library
    try:
        import toml
    except ImportError:
        print("❌ Missing 'toml' library.")
        print("Install it with: pip install toml")
        sys.exit(1)
    
    success = test_connection()
    if not success:
        print("\n🔧 Summary of what to check:")
        print("1. secrets.toml has [google_sheets] section")
        print("2. private_key has \\n characters (not actual newlines)")
        print("3. Sheet shared with service account")
        print("4. Service account enabled in Google Cloud")
        sys.exit(1)
