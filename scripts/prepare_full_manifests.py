"""Combine real and synthetic manifests into full development and test manifests."""
import csv
import os

def create_full_manifests():
    # 1. Development
    dev_rows = []
    # Real dev
    with open("data/manifest_development.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["images_dir"] = "data/development"
            r["mask_filename"] = ""
            r["x0"] = r["y0"] = r["x1"] = r["y1"] = r["x2"] = r["y2"] = r["x3"] = r["y3"] = ""
            dev_rows.append(r)
            
    # Synthetic dev
    with open("data/synthetic/manifest_synthetic.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "development":
                r["original_path"] = os.path.join("data/synthetic", r["filename"])
                r["images_dir"] = "data/synthetic"
                dev_rows.append(r)
                
    fieldnames = ["image_id", "split", "document_name", "condition", "is_positive", "filename", "original_path", "images_dir", "mask_filename", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]
    with open("data/manifest_development_full.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dev_rows)
        
    # 2. Test
    test_rows = []
    with open("data/manifest_test.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["images_dir"] = "data/test"
            r["mask_filename"] = ""
            r["x0"] = r["y0"] = r["x1"] = r["y1"] = r["x2"] = r["y2"] = r["x3"] = r["y3"] = ""
            test_rows.append(r)
            
    with open("data/synthetic/manifest_synthetic.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                r["original_path"] = os.path.join("data/synthetic", r["filename"])
                r["images_dir"] = "data/synthetic"
                test_rows.append(r)
                
    with open("data/manifest_test_full.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(test_rows)
        
    print(f"Created data/manifest_development_full.csv with {len(dev_rows)} images (36 real + 6 synthetic).")
    print(f"Created data/manifest_test_full.csv with {len(test_rows)} images (24 real + 6 synthetic).")

if __name__ == "__main__":
    create_full_manifests()
