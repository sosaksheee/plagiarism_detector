#  Plagiarism Detection System (TF-IDF + SBERT)

A **Python-based plagiarism detection system** that identifies **lexical and semantic plagiarism** in **text and source code** using **TF-IDF** and **Sentence-BERT (SBERT)**. The project supports both **CLI** and **GUI (Tkinter)** usage, making it suitable for **academic projects, demos, and real-world submissions**.

---

##  Features

*  Detects plagiarism in **text documents** and **programming code**
*  Uses **TF-IDF + Cosine Similarity** for lexical overlap
*  Uses **Sentence-BERT (SBERT)** for semantic plagiarism (paraphrasing detection)
*  Supports **file vs file** and **file vs folder** comparison
*  User-friendly **Tkinter GUI** (no command line required)
*  Generates **CSV plagiarism reports**
*  Lightweight model (no GPU required)

---

##  Technologies Used

* Python 3.9+
* scikit-learn
* sentence-transformers (SBERT)
* PyTorch
* pandas
* Tkinter

---

##  Supported File Types

| Type | Extensions                    |
| ---- | ----------------------------- |
| Text | `.txt`, `.md`                 |
| Code | `.py`, `.java`, `.cpp`, `.js` |

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/plagiarism-detection-system.git
cd plagiarism-detection-system
```

Install dependencies:

```bash
pip install sentence-transformers torch scikit-learn pandas
```

---

##  Usage

### Option 1: GUI Mode (Recommended)

Run:

```bash
python main.py
```

Steps:

1. Select **Query Document**
2. Select **Reference File or Folder**
3. Click **Run Plagiarism Check**
4. View results and generated CSV report

The report is saved as:

```
plagiarism_report_gui.csv
```

---

### Option 2: CLI Mode

```bash
python main.py --query submission.py --against previous_submissions/
```

Arguments:

* `--query` : Document to check
* `--against` : File or folder to compare against
* `--out` : Output CSV report (optional)

---

##  Output Example

| Reference Document | TF-IDF | SBERT | Verdict    |
| ------------------ | ------ | ----- | ---------- |
| student1.py        | 0.71   | 0.92  | Suspicious |
| student2.py        | 0.12   | 0.30  | Clean      |

---

##  Example Test Setup

```
project/
├── main.py
├── submission.py
└── previous_submissions/
    ├── student1.py
    ├── student2.py
```

---

##  Academic Relevance

This project is ideal for:

* College mini / major projects
* NLP coursework
* Academic plagiarism detection demos
* Viva and project evaluations

It demonstrates:

* Natural Language Processing
* Semantic similarity using Transformers
* Practical machine learning application

---

##  Future Improvements

* Sentence / line-level plagiarism highlighting
* AST-based code similarity
* PDF report generation
* Web-based interface
* Database-backed storage

---

##  Contributing

Pull requests are welcome. For major changes, please open an issue first.

---


##  Acknowledgements

* Sentence Transformers by UKP Lab
* scikit-learn community
* Open-source NLP ecosystem

---

⭐ If you find this project useful, please give it a star on GitHub!
