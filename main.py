import re
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


TEXT_EXTENSIONS = {".txt", ".md"}
CODE_EXTENSIONS = {".py", ".java", ".cpp", ".js"}

TFIDF_THRESHOLD = 0.75
SBERT_THRESHOLD = 0.80
SBERT_MODEL = "all-MiniLM-L6-v2"


# ---------------- Core Logic ----------------

def load_file(path: Path) -> Tuple[str, str]:
    return path.name, path.read_text(encoding="utf-8")


def load_references(path: Path) -> List[Tuple[str, str]]:
    if path.is_file():
        return [load_file(path)]

    docs = []
    for file in path.rglob("*"):
        if file.suffix.lower() in TEXT_EXTENSIONS | CODE_EXTENSIONS:
            docs.append(load_file(file))

    if not docs:
        raise ValueError("No valid reference documents found.")

    return docs


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_code(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"\s+", " ", code)
    return code.strip().lower()


def preprocess(name: str, content: str) -> str:
    return preprocess_code(content) if Path(name).suffix in CODE_EXTENSIONS else preprocess_text(content)


def compute_tfidf_similarity(docs: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    return cosine_similarity(tfidf)


def compute_sbert_similarity(docs: List[str]) -> np.ndarray:
    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    return cosine_similarity(embeddings)


def run_plagiarism(query: Tuple[str, str], references: List[Tuple[str, str]]) -> pd.DataFrame:
    all_docs = [query] + references
    processed = [preprocess(n, c) for n, c in all_docs]

    tfidf_sim = compute_tfidf_similarity(processed)
    sbert_sim = compute_sbert_similarity(processed)

    rows = []
    for i, (ref_name, _) in enumerate(references, start=1):
        tfidf_score = tfidf_sim[0][i]
        sbert_score = sbert_sim[0][i]

        verdict = "Suspicious" if (
            tfidf_score >= TFIDF_THRESHOLD or sbert_score >= SBERT_THRESHOLD
        ) else "Clean"

        rows.append({
            "Reference Document": ref_name,
            "TF-IDF": round(tfidf_score, 3),
            "SBERT": round(sbert_score, 3),
            "Verdict": verdict
        })

    return pd.DataFrame(rows).sort_values(by=["SBERT", "TF-IDF"], ascending=False)


# ---------------- GUI ----------------

class PlagiarismGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plagiarism Detection System")
        self.root.geometry("750x450")

        self.query_path: Path | None = None
        self.ref_path: Path | None = None

        self.build_ui()

    def build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        tk.Button(frame, text="Select Query Document", command=self.select_query).pack(fill="x")
        tk.Button(frame, text="Select Reference File / Folder", command=self.select_reference).pack(fill="x", pady=5)
        tk.Button(frame, text="Run Plagiarism Check", command=self.run).pack(fill="x", pady=10)

        self.tree = ttk.Treeview(frame, columns=("tfidf", "sbert", "verdict"), show="headings")
        self.tree.heading("tfidf", text="TF-IDF")
        self.tree.heading("sbert", text="SBERT")
        self.tree.heading("verdict", text="Verdict")
        self.tree.pack(fill="both", expand=True)

    def select_query(self) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.query_path = Path(path)
            messagebox.showinfo("Selected", f"Query:\n{self.query_path}")

    def select_reference(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            path = filedialog.askopenfilename()
        if path:
            self.ref_path = Path(path)
            messagebox.showinfo("Selected", f"Reference:\n{self.ref_path}")

    def run(self) -> None:
        if not self.query_path or not self.ref_path:
            messagebox.showerror("Error", "Please select both query and reference.")
            return

        try:
            query = load_file(self.query_path)
            refs = load_references(self.ref_path)
            report = run_plagiarism(query, refs)

            self.tree.delete(*self.tree.get_children())
            for _, row in report.iterrows():
                self.tree.insert("", "end", values=(
                    row["TF-IDF"], row["SBERT"], row["Verdict"]
                ))

            report.to_csv("plagiarism_report_gui.csv", index=False)
            messagebox.showinfo("Done", "Report saved as plagiarism_report_gui.csv")

        except Exception as e:
            messagebox.showerror("Error", str(e))


def main() -> None:
    root = tk.Tk()
    PlagiarismGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
