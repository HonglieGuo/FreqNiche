import urllib.request
import hashlib
import os

def download_and_hash(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    
    sha256_hash = hashlib.sha256()
    with open(output_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    h = sha256_hash.hexdigest()
    print(f"SHA-256: {h}")
    return h

if __name__ == "__main__":
    os.makedirs("data/gmt", exist_ok=True)
    
    hallmark_url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    go_bp_url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023"
    
    hallmark_hash = download_and_hash(hallmark_url, "data/gmt/MSigDB_Hallmark_2020.gmt")
    go_bp_hash = download_and_hash(go_bp_url, "data/gmt/GO_Biological_Process_2023.gmt")
    
    with open("data/gmt/gmt_hashes.txt", "w") as f:
        f.write(f"MSigDB_Hallmark_2020.gmt: {hallmark_hash}\n")
        f.write(f"GO_Biological_Process_2023.gmt: {go_bp_hash}\n")
    print("Done. Hashes saved to data/gmt/gmt_hashes.txt")
