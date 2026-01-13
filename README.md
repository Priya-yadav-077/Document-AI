
# [Document-AI]

## 🚀 How to Run the Code

This project requires two steps: Indexing and Querying.

### 1. Indexing (Required Setup)

You must run this command first to prepare the PDF data for searching.

```bash
python main.py --index
```

**Note:** Run the index command **every time** you change or add a PDF file.

#### Alternate Loader

In case of difficulty with library "pdf2image" due to missing root rights, fallback to alternate loader with "pypdf":

```bash
python main.py --index --alternate_loader --pdf_path <<path_to_local_pdf>>
```

### 2\. Querying

Once indexing is complete, you can ask questions using the following command:

```bash
python main.py --query "Your question here"
```

## ⚙️ GPU Requirement

This code assumes you are running on at least **one GPU**.

  * To change the required GPU index, modify the `devices` variable in `summarizer.py`.

-----

