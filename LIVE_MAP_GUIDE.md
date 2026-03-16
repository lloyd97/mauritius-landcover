# 🛰️ Mauritius Live Satellite Map Viewer - NOW RUNNING!

## ✅ Server Status: ACTIVE

**Your live satellite map viewer is now running!**

## 🌐 Access the Application

**Open your web browser and visit:**
```
http://localhost:5000
```

## 🎯 How to Use

### Three-Panel Interface:

1. **LEFT PANEL - Interactive Map**
   - Click anywhere on the map to select a location
   - The map shows Mauritius with OpenStreetMap base layer
   - You can zoom in/out and pan around

2. **MIDDLE PANEL - Sentinel-2 Satellite Image**
   - Shows the REAL satellite image from that location
   - True-color RGB composite
   - 10-meter resolution imagery

3. **RIGHT PANEL - Land Cover Classification**
   - Shows the SAME image with colored regions
   - Classified using the U-Net model
   - Color-coded by land cover type

### Controls:

- **"Classify Current View"** - Analyzes the current map center
- **"Switch to 2015 Baseline"** - Toggle between current (2024) and historical (2015) imagery

### Color Legend (Right Panel):

- 🔲 **Grey** = Roads
- 💧 **Blue** = Water/Rivers
- 🌲 **Dark Green** = Forest
- 🌾 **Light Green** = Plantation/Sugar Cane
- 🏢 **Brown** = Buildings
- 🏜️ **Tan** = Bare Land

## 📋 Current Mode

**DEMO MODE** - Using synthetic satellite imagery

The application is currently running in DEMO mode because Google Earth Engine 
is not authenticated. This means:
- ✅ Interface works perfectly
- ✅ Side-by-side comparison works
- ✅ Color classification works
- ⚠️ Uses demo/synthetic satellite images instead of real Sentinel-2 data

## 🔓 Enable Real Sentinel-2 Data

To use LIVE satellite imagery from Google Earth Engine:

1. **Authenticate Google Earth Engine:**
   ```bash
   earthengine authenticate
   ```

2. **Follow the prompts:**
   - A browser will open
   - Sign in with your Google account
   - Authorize Earth Engine
   - Copy the verification code
   - Paste it back in the terminal

3. **Restart the server:**
   - Stop the current server (Ctrl+C or close terminal)
   - Run again: `cd src/web && py live_map_app.py`

4. **Enjoy live satellite data!**
   - Click anywhere in Mauritius
   - Get real Sentinel-2 imagery
   - See actual land cover classification

## 🎨 What You'll See

### Example Workflow:

1. **Open http://localhost:5000**
2. **Click on Port Louis** (capital city)
3. **Left**: Interactive map with marker
4. **Middle**: Satellite image of Port Louis
5. **Right**: Classified image showing:
   - Blue water (harbor)
   - Brown buildings (urban areas)
   - Grey roads (street network)
   - Green plantation (surrounding areas)

## 🚀 Features

✅ **Click-to-Classify** - Click any location to analyze  
✅ **Side-by-Side View** - Original vs Classified  
✅ **Time Comparison** - 2024 vs 2015 baseline  
✅ **Interactive Map** - Zoom and pan  
✅ **Color-Coded Legend** - Clear class identification  
✅ **Real-time Processing** - Instant classification  

## 🔧 Technical Details

- **Backend**: Flask + PyTorch
- **Frontend**: Leaflet.js interactive maps
- **Model**: U-Net with ResNet34 encoder
- **Input**: 9 channels (6 Sentinel-2 bands + 3 indices)
- **Output**: 7 land cover classes
- **Resolution**: 10m per pixel

## 📝 Next Steps

1. **Try it out!** - Open http://localhost:5000
2. **Click around Mauritius** - Try different locations
3. **Toggle time periods** - Compare 2024 vs 2015
4. **Authenticate GEE** - Get real satellite data

Enjoy your live satellite map viewer! 🇲🇺
