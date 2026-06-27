import os
import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset

def download_pubmed_subset(target_count: int = 100, output_dir: str = "data/raw"):
    """
    Downloads a subset of PubMed abstracts.
    Prioritizes PMIDs found in the PubMedQA dataset to ensure evaluation overlap,
    then fills the rest with random samples.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, f"pubmed_subset_{target_count}.jsonl")
    
    print(f"--- Step 1: Loading PubMedQA dataset to extract target PMIDs ---")
    # Correct namespace for PubMedQA
    pubmed_qa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    target_pmids = set(str(item['pubid']) for item in pubmed_qa)
    print(f"Found {len(target_pmids)} unique PMIDs in PubMedQA.")
    
    print(f"\n--- Step 2: Streaming scientific_papers/pubmed dataset ---")
    # Using the canonical scientific_papers dataset
    pubmed_stream = load_dataset("scientific_papers", "pubmed", split="train", streaming=True, trust_remote_code=True)
    
    matched_docs = []
    random_docs = []
    matched_pmids = set()
    
    for doc in pubmed_stream:
        # In scientific_papers, the PubMed ID is stored in 'article_id'
        doc_id = str(doc.get('article_id'))
        
        # Priority 1: Catch PubMedQA PMIDs
        if doc_id in target_pmids and doc_id not in matched_pmids:
            matched_docs.append(doc)
            matched_pmids.add(doc_id)
            if len(matched_docs) % 50 == 0:
                print(f"Matched PubMedQA PMIDs: {len(matched_docs)}/{len(target_pmids)}")
        else:
            # Priority 2: Fill up the remaining slots with random samples
            needed_random = target_count - len(matched_docs)
            if needed_random > 0 and len(random_docs) < needed_random:
                random_docs.append(doc)
                
        # Stop early if we have everything we need
        if len(matched_docs) == len(target_pmids) and len(matched_docs) + len(random_docs) >= target_count:
            print("Target counts reached. Stopping stream early.")
            break
            
    # Combine and shuffle
    print(f"\n--- Step 3: Finalizing subset ---")
    if len(matched_docs) >= target_count:
        random.shuffle(matched_docs)
        final_subset = matched_docs[:target_count]
    else:
        random.shuffle(random_docs)
        final_subset = matched_docs + random_docs[:target_count - len(matched_docs)]

    print(f"Saving {len(final_subset)} documents to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in final_subset:
            # Standardize keys so the chunker knows exactly what to look for
            # scientific_papers stores the abstract as a list of strings, so we join them
            abstract = doc.get("abstract", [])
            if isinstance(abstract, list):
                abstract = " ".join(abstract)
                
            standardized_doc = {
                "pubmed_id": doc.get("article_id"),
                "title": doc.get("title", ""),
                "abstract": abstract
            }
            f.write(json.dumps(standardized_doc) + '\n')
            
    print(f"✅ Download complete! File saved at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PubMed subset for POC")
    parser.add_argument("--count", type=int, default=100, help="Target number of documents (default: 100)")
    args = parser.parse_args()
    
    download_pubmed_subset(target_count=args.count)
