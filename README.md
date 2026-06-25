# minimal-lap-time

This project targets minimum-lap-time optimization for a racecar on a local karting track.
It is part of the lecture *Numerical Optimization* by Prof. Moritz Diehl in the summer term 2026 at the University of Freiburg.

## Python setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```


## Report compilation

The report is written in LaTeX. To compile it locally, a LaTeX distribution with `latexmk` is required.

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install texlive-latex-extra latexmk
```

### macOS

Install MacTeX using Homebrew:

```bash
brew install --cask mactex
```

Alternatively, download and install MacTeX from https://www.tug.org/mactex/.

### Windows

Install either:

- MiKTeX: https://miktex.org/
- TeX Live: https://www.tug.org/texlive/

After installation, verify that `latexmk` is available:

```bash
latexmk --version
```

### Compile the report

From the repository root:

```bash
cd report
latexmk -pdf main.tex
```

To remove auxiliary files generated during compilation:

```bash
latexmk -c
```