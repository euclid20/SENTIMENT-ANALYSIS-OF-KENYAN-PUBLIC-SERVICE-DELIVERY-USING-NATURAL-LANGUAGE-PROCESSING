import os

source_folder = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Transport dataset"

print(f"Checking directory: {source_folder}\n")

if not os.path.exists(source_folder):
    print("❌ ERROR: Python cannot even find this folder! Double-check the path layout.")
else:
    print("✅ Folder found! Listing all contents inside:")
    files = os.listdir(source_folder)
    
    if not files:
        print("   (The folder appears completely empty to Python)")
    else:
        for i, file in enumerate(files, 1):
            print(f"   {i}. {file}")