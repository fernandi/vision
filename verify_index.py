import os
import faiss
import json
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

DATA_DIR = "data/images"
INDEX_FILE = "data/faiss.index"
MAPPING_FILE = "data/index_mapping.json"
MODEL_ID = "openai/clip-vit-base-patch32"

def main():
    print("Loading resources...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load items
    index = faiss.read_index(INDEX_FILE)
    print(f"Index size: {index.ntotal}")
    
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
    print(f"Mapping size: {len(mapping)}")
    
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    # Test 1: Search for "landscape"
    print("\n--- Test 1: Text Search 'landscape' ---")
    inputs = processor(text=["landscape"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        if not isinstance(outputs, torch.Tensor):
             if hasattr(outputs, 'pooler_output'):
                 # It seems in some versions get_text_features might return the output of the text tower?
                 # Whatever it is, if it's not a tensor, let's try to get the tensor.
                 # But get_text_features is supposed to return the pooled and projected features.
                 # If it returns BaseModelOutputWithPooling, maybe we just need the pooler_output or last_hidden_state?
                 # Actually, let's just inspect what it is if it fails.
                 # But for now, let's assume it has pooler_output and that needs projection?
                 # OR, let's just use model(...) (forward) which is safer.
                 pass
        
        # Safer text embedding extraction using the model forward for text-only? 
        # Actually, let's just stick to the search_engine.py logic exactly:
        if not isinstance(outputs, torch.Tensor):
             if hasattr(outputs, 'pooler_output'):
                 # MIMIC SERVER LOGIC:
                 print("DEBUG: Applying model.text_projection to pooler_output (Server Logic)...")
                 text_features = model.text_projection(outputs.pooler_output)
             else:
                 text_features = outputs
        else:
             text_features = outputs

        # Double check it is a tensor now
        if not isinstance(text_features, torch.Tensor):
             # It might be in 'last_hidden_state' if pooler_output absent?
             if hasattr(text_features, 'last_hidden_state'):
                  text_features = text_features.last_hidden_state[:, 0, :] # CLS token

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_features.cpu().numpy().astype('float32')
        
    D, I = index.search(text_emb, 5)
    for i, idx in enumerate(I[0]):
        if idx != -1:
            item = mapping[idx]
            print(f"Rank {i+1}: ID={idx}, File={item.get('filename')}, Dist={D[0][i]:.4f}, Title={item.get('Title')}")

    # Test 2: Identity Search (Image 46.jpg - "Autumn Landscape")
    target_filename = "46.jpg"
    print(f"\n--- Test 2: Identity Search for {target_filename} ---")
    
    img_path = os.path.join(DATA_DIR, target_filename)
    if not os.path.exists(img_path):
        print(f"File {target_filename} not found!")
        return

    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if 'pixel_values' in inputs:
             img_inputs = inputs['pixel_values']
        else:
             img_inputs = inputs
        
        outputs = model.get_image_features(pixel_values=img_inputs)
        
        if not isinstance(outputs, torch.Tensor):
             if hasattr(outputs, 'pooler_output'):
                 image_features = outputs.pooler_output
             else:
                 image_features = outputs
        else:
             image_features = outputs
             
        # Double check it is a tensor now
        if not isinstance(image_features, torch.Tensor):
             # It might be in 'last_hidden_state' if pooler_output absent?
             if hasattr(image_features, 'last_hidden_state'):
                  image_features = image_features.last_hidden_state[:, 0, :] # CLS token

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        img_emb = image_features.cpu().numpy().astype('float32')

    D, I = index.search(img_emb, 5)
    
    found_self = False
    for i, idx in enumerate(I[0]):
        if idx != -1:
            item = mapping[idx]
            is_match = item.get('filename') == target_filename
            prefix = "MATCH!" if is_match else ""
            if is_match: found_self = True
            print(f"Rank {i+1}: {prefix} ID={idx}, File={item.get('filename')}, Dist={D[0][i]:.4f}, Title={item.get('Title')}")

    if found_self:
        print("\nSUCCESS: Image found itself in search results.")
    else:
        print("\nFAILURE: Image did NOT find itself. Index is likely misaligned or embeddings are broken.")

if __name__ == "__main__":
    main()
