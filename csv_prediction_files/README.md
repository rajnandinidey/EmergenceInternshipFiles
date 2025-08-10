# Data Folder for CSV Files

This folder is designed to store CSV data files for the machine learning project.

## 📁 How to Upload CSV Files

### Method 1: GitHub Web Interface (Recommended)
1. Navigate to this folder on GitHub
2. Click the **"Add file"** button (dropdown arrow)
3. Select **"Upload files"**
4. Drag and drop your CSV files or click to browse
5. Add a commit message (e.g., "Add dataset files")
6. Click **"Commit changes"**

### Method 2: Git Commands (Advanced)
```bash
# Copy your CSV files to this folder locally
cp /path/to/your/file.csv data/

# Add and commit
git add data/*.csv
git commit -m "Add new CSV datasets"
git push
```

## 📊 Supported File Types
- `.csv` - Comma-separated values
- `.txt` - Tab-delimited text files
- `.xlsx` - Excel files (will be converted to CSV)

## ⚠️ Important Notes
- Keep file sizes under 100MB for GitHub
- Use descriptive filenames
- Consider compressing large datasets
- Update this README when adding new data sources

## 🔍 Current Contents
This folder is ready to receive your CSV files!
