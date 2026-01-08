# Google Earth Engine Setup

## The Issue
GEE needs a project ID to work. You need to either:
1. Create a Google Cloud Project, OR
2. Use Earth Engine's default project

## Quick Fix - Use Your GEE Account

Run this command:
```bash
earthengine set_project YOUR_PROJECT_ID
```

## How to Get Your Project ID:

### Option 1: Use Earth Engine (Recommended)
1. Go to: https://code.earthengine.google.com/
2. Sign in with your Google account
3. Look at the top - you'll see your project name
4. Use that project ID

### Option 2: Create a New Project
1. Go to: https://console.cloud.google.com/
2. Create a new project (free)
3. Enable Earth Engine API
4. Copy the project ID

## Then Run:
```bash
# Replace 'your-project-id' with your actual project ID
earthengine set_project your-project-id

# Or for a quick test, many people use:
earthengine set_project ee-lloydsproject
```

## Alternative: Set in Code
Add this to your environment or code:
```bash
export EARTHENGINE_PROJECT=your-project-id
```

Once set, re-run the download script!
