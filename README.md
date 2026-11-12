# DDoS Detection Paper Experiment

---

## Step 1: Download the dataset from Kaggle

**Link:** https://www.kaggle.com/datasets/dhoogla/cicddos2019

1. Open the link and sign in. A free Kaggle account works, and Google sign-in is fine.
2. Click **Download** (top right), then **Download dataset as zip**.
3. You'll get a file called **`archive.zip`**. You don't need to unzip it.

This is a cleaned, much smaller copy of CIC-DDoS2019 in Parquet format. Files with `-training` in the name are day 1, and files with `-testing` in the name are day 2. The script sorts them automatically.

## Step 2: Run the experiment (choose where)

### Option 1: On your own computer (simplest after downloading)

1. Install Python 3 (https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during install.
2. Put `ddos_experiment.py` and `archive.zip` **in the same folder**.
3. Open a terminal in that folder. On Windows, click the folder's address bar, type `cmd` and press Enter.
4. Install the libraries (one time only):
```
pip install pandas pyarrow scikit-learn matplotlib
```
5. Run:
```
python ddos_experiment.py --data archive.zip
```
On Windows, if `python` isn't found, use `py ddos_experiment.py --data archive.zip`.

It runs in a few minutes and creates a **`results`** folder next to the script.

### Option 2: In Google Colab (if your computer is slow)

1. Upload `ddos_experiment.py` and `archive.zip` to your Google Drive, for example into `MyDrive/ddos/`.
2. In a new Colab notebook (https://colab.research.google.com), run:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ddos
!python ddos_experiment.py --data archive.zip
```
The `results` folder will appear in `MyDrive/ddos/`.

### Reading your results
Open **`results/results_summary.txt`** first. It lists every number used in the paper.

**Note:** this Kaggle copy appears to have no timestamp column. If so, the script prints a note and skips the measured window analysis. The paper then automatically uses a calculated version of Section VII-C and Fig. 4. **You don't need to change anything.**

(If you ever use the original UNB CSV files instead, run `python ddos_experiment.py --day1 CSV-01-12 --day2 CSV-03-11`.)

## Step 3: The images and files

After the run, `results/` will contain:

| File | Where it goes in the paper |
|---|---|
| `fig_split_comparison.png` | **Fig. 2**: random split vs next day |
| `fig_per_attack_recall.png` | **Fig. 3**: recall per attack type |
| `fig_window_tradeoff.png` | **Fig. 4**, only if your data has timestamps. Otherwise the paper draws Fig. 4 itself. |
| `fig_feature_importance.png` | **Fig. 5**: Appendix A, top features |
| `results_table.tex` | Rows of **Table III** (filled in automatically) |
| `paper_values.tex` | All numbers in the text (filled in automatically) |
| `*.csv`, `results_summary.txt` | Your raw numbers, for checking and for your own understanding |
