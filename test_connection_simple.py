"""
SIMPLE Google Sheets Test - Reads from local file only
"""

import os
import sys

print("🔍 Simple Google Sheets Test")
print("=" * 60)

# Check for secrets file
secrets_file = ".streamlit/secrets.toml"
if not os.path.exists(secrets_file):
    print(f"❌ ERROR: File not found: {secrets_file}")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)

print(f"✅ Found: {secrets_file}")

# Try to read it as plain text first
try:
    with open(secrets_file, 'r') as f:
        content = f.read()
    
    print("✅ File can be read")
    
    # Quick check for [google_sheets]
    if "[google_sheets]" not in content:
        print("❌ ERROR: File doesn't contain [google_sheets] section")
        print("First 200 chars of file:")
        print(content[:200])
        sys.exit(1)
    
    print("✅ Found [google_sheets] section")
    
except Exception as e:
    print(f"❌ Can't read file: {e}")
    sys.exit(1)

# Now try to load as TOML
try:
    import toml
    secrets = toml.load(secrets_file)
    
    if "google_sheets" not in secrets:
        print("❌ ERROR: No 'google_sheets' key in TOML")
        print("File keys:", list(secrets.keys()))
        sys.exit(1)
    
    creds = secrets["google_sheets"]
    print(f"✅ TOML parsed successfully")
    print(f"   Project: {creds.get('project_id', 'NOT FOUND')}")
    print(f"   Account: {creds.get('client_email', 'NOT FOUND')}")
    
except ImportError:
    print("❌ Missing 'toml' library")
    print("Install with: pip install toml")
    sys.exit(1)
except Exception as e:
    print(f"❌ TOML parsing error: {e}")
    sys.exit(1)

# Test Google connection
print("\n🔗 Testing Google connection...")
try:
    from google.oauth2.service_account import Credentials
    import gspread
    
    credentials = Credentials.from_service_account_info(
        creds,
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    
    client = gspread.authorize(credentials)
    print("✅ Google authentication successful!")
    
    # Try to open the sheet
    sheet_url = "https://docs.google.com/spreadsheets/d/13Zw4TksoH9P1PWv1HuNpapZOOnT_dIm_Pr85qRof4yE"
    spreadsheet = client.open_by_url(sheet_url)
    print(f"✅ Sheet found: {spreadsheet.title}")
    
    # Test write
    worksheet = spreadsheet.sheet1
    import datetime
    
    test_row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "TEST", "TEST", "SIMPLE TEST", "TEST", "1x", "High",
        "PENDING", "", "", "Simple connection test"
    ]
    
    worksheet.append_row(test_row)
    print("✅ Write test successful!")
    
    # Clean up
    all_data = worksheet.get_all_values()
    if len(all_data) > 1 and "SIMPLE TEST" in all_data[-1][3]:
        worksheet.delete_rows(len(all_data))
        print("✅ Cleaned up test row")
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Everything works!")
    print("Your app should now save predictions to Google Sheets.")
    
except Exception as e:
    error_msg = str(e)
    print(f"❌ Google Sheets error: {error_msg}")
    
    if "invalid_grant" in error_msg.lower():
        print("\n🔍 ISSUE: Invalid JWT Signature")
        print("Check your private_key format in secrets.toml")
        print("It should have \\n characters, not actual newlines")
        
    elif "PERMISSION_DENIED" in error_msg:
        print("\n🔍 ISSUE: Permission Denied")
        print(f"Share your Google Sheet with:")
        print(f"  {creds.get('client_email', 'UNKNOWN')}")
        
    elif "SpreadsheetNotFound" in error_msg:
        print("\n🔍 ISSUE: Sheet not found")
        print("Check the URL in your code")
        
    sys.exit(1)
