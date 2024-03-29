# Automating flashcards extraction for language learning

This repository implements a workflow to extract unique words from a PDF and output them in a flashcard format, compatible with Anki or Quizlet.

See this X (Twitter) post for the story behind this repository, and follow me for more updates!
https://twitter.com/AntonObukhov1/status/1758640714613932321

## Begegnungen B1

### A NOTE from 2024.Feb.21: 
Bad cards exist with wrong spelling (rechzeitig), the wrong first side (plural instead of singular), and maybe worse, e.g., incorrect definite article, reflectivity, and tense forms. Use at your own risk! If you feel like contributing back, please create an issue [here](https://github.com/toshas/language-learning/issues) with the title "[DECK NAME] [CARD NAME]" and explain the problem in the issue body. 

## Setup

(Linux, Mac, Windows WSL)

```shell
ENV_PATH=~/venv_flashcards
python3 -m venv $ENV_PATH
source $ENV_PATH/bin/activate 
pip3 install -r requirements.txt
```

## OpenAI API Key

1. Make one here: https://platform.openai.com/api-keys
2. Place it into the project root named as a `openai_api_key.txt` file before running the script.

## Example usage

```shell
python3 language_cards.py \
    <PATH-TO-PDF> \
    --chapter_pages 9 39 73 101 133 161 189 217 245 \
    --kill Deutschland Hessen Ihr ab aufraumen "das Cafe" "das Goethezimmer" geschenkt \
    --strip " - added for completeness, already marked as duplicate" ", ignore" "- duplicate" " Duplicate" " [Duplicate]"
```

## Command line options

```shell
python3 language_cards.py --help
usage: language_cards.py [-h] [--separator SEPARATOR] [--pdf_page_first PDF_PAGE_FIRST] [--pdf_page_last PDF_PAGE_LAST] [--out_path OUT_PATH] [--resolution RESOLUTION] [--chapter_pages CHAPTER_PAGES [CHAPTER_PAGES ...]] [--kill KILL [KILL ...]] [--strip STRIP [STRIP ...]] [--verbose] pdf_path

Extract language cards from a PDF

positional arguments:
  pdf_path              PDF input

options:
  -h, --help            show this help message and exit
  --separator SEPARATOR
                        Separator character. = for Quizlet, for Anki
  --pdf_page_first PDF_PAGE_FIRST
                        First page to process
  --pdf_page_last PDF_PAGE_LAST
                        Last page to process
  --out_path OUT_PATH   Output directory path
  --resolution RESOLUTION
                        Optical recognition resolution
  --chapter_pages CHAPTER_PAGES [CHAPTER_PAGES ...]
                        List of page numbers with new chapters
  --kill KILL [KILL ...]
                        Cards keys to remove from the output
  --strip STRIP [STRIP ...]
                        Texts to strip from cards
  --verbose             Output more details about the process
```

- `separator` lets you change the separator character, separating the front and back sides content within each line. Default: tab (Anki).
- `pdf_page_first` and `pdf_page_last` can specify the range of pages from the PDF file to process. Default: full PDF.
- `out_path` can override the default output location. Default: <PDF-PATH>_output.
- `resolution` specifies the resolution of optical recognition. Default: 1024.
- `chapter_pages` allows to split output into separate files, corresponding to the book chapters. Page numbers are the corresponding chapters' first page numbers. Important: the page numbers should be checked in a PDF viewer, they often do not match the page numbers printed inside books.
- `kill` lets the user remove certain entries from the output (usually during the iterative process).
- `strip` lets the user clean up any remaining verbosity from the LLM.
- `verbose` makes the analysis a bit more verbose (useful for debugging).

## License

Apache License, Version 2.0
