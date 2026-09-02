import json
import codecs

with open('sst_metadata.json', 'rb') as f:
    raw = f.read()

# Try to decode safely
try:
    if raw.startswith(codecs.BOM_UTF16_LE):
        text = raw[len(codecs.BOM_UTF16_LE):].decode('utf-16le')
    elif raw.startswith(codecs.BOM_UTF16_BE):
        text = raw[len(codecs.BOM_UTF16_BE):].decode('utf-16be')
    else:
        text = raw.decode('utf-8')
        
    data = json.loads(text)
    
    if isinstance(data, list):
        for item in data:
            print("Product ID:", item.get('id', 'N/A'))
            datasets = item.get('datasets', [])
            print("Dataset IDs:")
            for d in datasets:
                print(f"  - {d.get('id')}")
    elif isinstance(data, dict):
        print("Product ID:", data.get('id', 'N/A'))
        datasets = data.get('datasets', [])
        print("Dataset IDs:")
        for d in datasets:
            print(f"  - {d.get('id')}")
            
except Exception as e:
    print("Error parsing:", e)
