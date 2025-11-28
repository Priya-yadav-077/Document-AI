# main.py -entry point 
import argparse
from rag_pipeline import setup_retriever, query_rag

def main():
    parser = argparse.ArgumentParser(description="Local Multimodal RAG (single-PDF).")
    parser.add_argument("--index", action="store_true", help="Build the index from the PDF.")
    parser.add_argument("--query", type=str, help="Ask a question to the indexed document.")
    args = parser.parse_args()

    if args.index:
        print("Indexing document (this may take a few minutes)...")
        setup_retriever() 
        print("Indexing complete. You can now run queries with --query.")
        return

    if args.query:
        print(f"Querying for: {args.query}")
        out = query_rag(args.query)
        print("\n--- ANSWER ---\n")
        print(out["response"])
        print("\n--- CONTEXT USED ---\n")
        for i, txt in enumerate(out["context"]["texts"]):
            print(f"[{i+1}] {txt[:500]}...\n")
        if out["context"]["images"]:
            print("Images were used in context (base64 strings). Use a Jupyter environment to display them.")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
