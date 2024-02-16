import os
import sys
import re

from PIL import Image
from pdf2image import convert_from_path
import requests

import pytesseract
from tqdm import tqdm


def get_prompt(ocr_text):
    prompt = (
        "I want to make word cards to learn German from the input text attached at the very end of this message after "
        "the percentage sign. "
        "Please extract all German words (only nouns, verbs, adjectives, and adverbs) from the input text, resulting "
        "from the tesseract OCR. "
        "Some words are merged together due to incorrectly recognized whitespace - please extract them too. "
        "After the extraction, normalize them according to the rules below. "
        "After normalization, keep only unique normalized words and output only one line per unique word according to "
        "the rules below. "
        'A noun should be normalized to its singular form in Nominative, e.g., "Bücher" should become "das Buch". '
        "Output each unique normalized noun and provide an English and Russian translation, along with the definite "
        "article and the plural form. "
        'Example format: "# das Buch = book, книга (die Bücher)". '
        'A verb should be converted to infinitive form in present, e.g., "ging" should become "gehen". '
        'In the example sentence "Räumst du auf?", extract the separable prefix verb as "aufraumen". '
        "Output each unique normalized verb with an English and Russian translation, and include the simple past and "
        "perfect participle forms. "
        'If used verb is transitive in the input text, please prepend "sich" in front of the verb in the output. '
        'Example format: "# aufräumen = to tidy up, убираться (räumte auf, habe aufgeräumt)". '
        'Another example format: "# sich rasieren = shave oneself, бриться (rasierte, habe rasiert)". '
        "An adjective should be normalized to its positive base form, removing comparative and superlative endings, "
        "ignoring inflections for case, gender, and number. "
        "This should result in the form that would appear before a masculine nominative singlular noun, e.g., "
        '"am schnellsten" should become "schnell". '
        "Output each unique normalized adjective with an English and Russian translation, and include the comparative "
        'and superlative forms. Example format: "# schnell = fast, быстрый (schneller, am schnellsten)". '
        "Since adverb normalization is typically a no-operation due to their invariable form, output them directly. "
        "However, ensure to provide an English and Russian translation. "
        'Example format for a unique normalized adverb: "# gerne = gladly, любезно". '
        "Do not output anything other than formatted lines, corresponding to nouns, verbs, adjectives, and adverbs "
        "extracted from the input text. "
        "Do not output the given examples, do not output words not from the input text, and do not miss words which "
        "are present in the input text. "
        "Here goes the input text after the percentage sign. "
        "% "
    )
    return prompt + ocr_text


def get_openai_api_key():
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is not None:
        return openai_api_key
    candidate_path = os.path.join(os.path.dirname(__file__), "openai_api_key.txt")
    if os.path.exists(candidate_path):
        with open(candidate_path, "r") as fp:
            openai_api_key = fp.read().strip()
        if len(openai_api_key) > 0:
            return openai_api_key
        else:
            raise RuntimeError("OpenAI API Key not found")
    else:
        raise RuntimeError("OpenAI API Key not found")


def extract_cards_one_page(
    ocr_text, openai_api_key, openai_api_max_tokens=2048, num_inferences=1
):
    headers = {  # fmt: skip
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    payload = {
        "model": "gpt-4-0125-preview",
        "messages": [
            {
                "role": "user",
                "content": get_prompt(ocr_text),
            }
        ],
        "max_tokens": openai_api_max_tokens,
        "n": num_inferences,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()


def parse_pdf_into_pages(
    path_pdf_input,
    path_output,
    first_page=None,
    last_page=None,
    longer_side=2048,
    thread_count=8,
):
    return convert_from_path(
        path_pdf_input,
        first_page=first_page,
        last_page=last_page,
        output_folder=path_output,
        fmt="png",
        thread_count=thread_count,
        size=longer_side,
        output_file="page",
        paths_only=True,
    )


def process_pdf_raw(
    path_pdf_input,
    path_output,
    first_page=None,
    last_page=None,
):
    print("Extracting PDF pages...")
    path_output_pages_raw = os.path.join(path_output, "pages_raw")
    marker_complete = os.path.join(path_output_pages_raw, ".marker.complete")
    if not os.path.exists(marker_complete):
        os.makedirs(path_output_pages_raw, exist_ok=True)
        list_page_files = parse_pdf_into_pages(
            path_pdf_input,
            path_output_pages_raw,
            first_page=first_page,
            last_page=last_page,
        )
        for page_file in list_page_files:
            basename = os.path.basename(page_file)
            baseid = os.path.splitext(basename)[0].split("-")[-1]
            new_path = os.path.join(path_output_pages_raw, "page" + baseid + ".png")
            os.rename(page_file, new_path)
        with open(marker_complete, "w") as _:
            pass
    return path_output_pages_raw


def compute_page_dimensions(width, height, longest_side):
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = longest_side
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = longest_side
        new_width = int(new_height * aspect_ratio)
    return new_width, new_height


def resize_pages(path_output, longest_side=1024):
    print("Resizing pages...")
    path_output_pages_raw = os.path.join(path_output, "pages_raw")
    path_output_pages_resized = os.path.join(path_output, "pages_resized")
    marker_complete = os.path.join(path_output_pages_resized, ".marker.complete")
    if not os.path.exists(marker_complete):
        os.makedirs(path_output_pages_resized, exist_ok=True)
        for page_file in os.listdir(path_output_pages_raw):
            if not page_file.endswith(".png"):
                continue
            path_in = os.path.join(path_output_pages_raw, page_file)
            path_out = os.path.join(path_output_pages_resized, page_file)
            img_in = Image.open(path_in)
            new_w, new_h = compute_page_dimensions(
                img_in.width, img_in.height, longest_side
            )
            img_in.resize((new_w, new_h), resample=Image.Resampling.LANCZOS).save(
                path_out
            )
        with open(marker_complete, "w") as _:
            pass
    return path_output_pages_resized


def filter_responses(data, num_inferences=3, verbose=False):
    out = {}
    counts = {}
    for lines in data:
        unique_lines = {}
        for line in lines.split("\n"):
            if not line.startswith("# "):
                if verbose:
                    print(f"Skipping: {line}")
                continue
            line = line[2:]
            parts = line.split("=")
            entry = parts[0].strip()
            if len(entry) == 0:
                if verbose:
                    print(f"Malformed line: {line}")
                continue
            if entry not in unique_lines:
                unique_lines[entry] = line
            if len(unique_lines[entry]) < len(line):
                unique_lines[entry] = line
        for k, v in unique_lines.items():
            if k not in out:
                out[k] = v
                counts[k] = 1
            else:
                counts[k] += 1
                if len(out[k]) < len(v):
                    out[k] = v
    assert num_inferences % 2 == 1
    min_count = (num_inferences + 1) // 2
    for k, c in counts.items():
        if c < min_count:
            del out[k]
    out = dict(sorted(out.items()))
    return out


def ocr_one_page(path_page):
    verbatim_output = pytesseract.image_to_string(Image.open(path_page), lang="deu")
    cleaned_text = re.sub(
        r"[0-9]|[\]{}©_]|[\n\r]+", " ", verbatim_output
    )  # Remove unwanted characters and digits
    cleaned_text = re.sub(
        " +", " ", cleaned_text
    )  # Replace multiple spaces with a single space
    return cleaned_text


def ocr_all_pages(path_output):
    print("Performing OCR...")
    path_output_pages_raw = os.path.join(path_output, "pages_resized")
    path_output_pages_ocr = os.path.join(path_output, "pages_ocr")
    marker_complete = os.path.join(path_output_pages_ocr, ".marker.complete")
    if not os.path.exists(marker_complete):
        os.makedirs(path_output_pages_ocr, exist_ok=True)
        for page_file in tqdm(sorted(os.listdir(path_output_pages_raw))):
            if not page_file.endswith(".png"):
                continue
            path_in = os.path.join(path_output_pages_raw, page_file)
            path_out = os.path.join(
                path_output_pages_ocr, os.path.splitext(page_file)[0] + ".txt"
            )
            text_out = ocr_one_page(path_in)
            with open(path_out, "w") as fp:
                fp.write(text_out)
        with open(marker_complete, "w") as _:
            pass
    return path_output_pages_ocr


def extract_cards_all_pages(path_output, num_inferences=3, verbose=False):
    assert num_inferences % 2 == 1
    print("Extracting cards with GPT...")

    path_output_pages_ocr = os.path.join(path_output, "pages_ocr")
    path_output_cards = os.path.join(path_output, "cards_pages_independent")
    marker_complete = os.path.join(path_output_cards, ".marker.complete")
    token_cost = 0
    if not os.path.exists(marker_complete):
        os.makedirs(path_output_cards, exist_ok=True)
        pbar = tqdm(sorted(os.listdir(path_output_pages_ocr)))
        for page_file in pbar:
            if not page_file.endswith(".txt"):
                continue
            path_in = os.path.join(path_output_pages_ocr, page_file)
            path_out_txt = os.path.join(path_output_cards, page_file)
            path_out_json = os.path.splitext(path_out_txt)[0] + ".json"

            if verbose:
                print("Checking ", path_out_txt)

            if os.path.exists(path_out_txt):
                continue

            with open(path_in, "r") as fp:
                ocr_text = fp.read()

            data = extract_cards_one_page(
                ocr_text, openai_api_key, num_inferences=num_inferences
            )
            with open(path_out_json + ".tmp", "w") as fp:
                fp.write(str(data))
            os.rename(path_out_json + ".tmp", path_out_json)

            try:
                cost = data["usage"]["total_tokens"]
                data = [
                    data["choices"][i]["message"]["content"]
                    for i in range(num_inferences)
                ]
                token_cost += cost
                pbar.set_description(f"TokenCost: {token_cost}")
            except Exception as e:
                print(f"\n===== ERROR processing {page_file}, text:\n{ocr_text}\n=====")
                print(data)
                print(e)
                exit(1)
            data = filter_responses(data, num_inferences, verbose=verbose)
            text_lines = "\n".join(data.values())

            with open(path_out_txt + ".tmp", "w") as fp:
                fp.write(text_lines)
            os.rename(path_out_txt + ".tmp", path_out_txt)
        with open(marker_complete, "w") as _:
            pass
    return path_output_cards


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Expecting at least a path to the PDF file"
    path_pdf_input = sys.argv[1]
    if len(sys.argv) >= 3:
        path_output = sys.argv[2]
    else:
        path_output = os.path.splitext(path_pdf_input)[0] + "_output"

    verbose = False
    longest_side = 1024

    openai_api_key = get_openai_api_key()

    path_output_pages_raw = process_pdf_raw(
        path_pdf_input, path_output, first_page=None, last_page=None
    )
    path_output_pages_resized = resize_pages(path_output, longest_side=longest_side)
    path_output_ocr = ocr_all_pages(path_output)
    path_output_cards = extract_cards_all_pages(path_output, verbose=verbose)
    print("Done.")
