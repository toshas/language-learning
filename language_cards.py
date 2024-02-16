import argparse
import os
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
        'In the example sentence "Räumst du auf?", extract the separable prefix verb as "aufraümen". '
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
    pdf_path,
    path_output,
    first_page=None,
    last_page=None,
    longer_side=2048,
    thread_count=8,
):
    return convert_from_path(
        pdf_path,
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
    pdf_path,
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
            pdf_path,
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
            if len(unique_lines[entry]) < len(line):  # choose a longer flip side
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


def group_chapters(path_output, chapter_pages, kill_list, strip_list):
    print("Grouping cards by chapters...")

    path_input_cards = os.path.join(path_output, "cards_pages_independent")
    if chapter_pages is None:
        path_output_by_chapter = os.path.join(path_output, "cards_final")
        path_output_cumulative = None
    else:
        path_output_by_chapter = os.path.join(path_output, "cards_final_by_chapter")
        path_output_cumulative = os.path.join(path_output, "cards_final_cumulative")
    marker_complete = os.path.join(path_output_by_chapter, ".marker.complete")

    chapter_cards = {}
    cumulative_cards = {}
    chapter_id = 0

    def dump_chapter():
        nonlocal chapter_cards, cumulative_cards, chapter_id
        chapter_file = f"chapter{chapter_id:02d}.txt"

        if path_output_cumulative is not None:
            for k in list(chapter_cards.keys()):
                if k not in cumulative_cards:
                    cumulative_cards[k] = chapter_cards[k]
                else:
                    del chapter_cards[k]

            cumulative_cards = dict(sorted(cumulative_cards.items()))
            with open(os.path.join(path_output_cumulative, chapter_file), "w") as fp:
                fp.write("\n".join([f"{k} = {v}" for k, v in cumulative_cards.items()]))

        chapter_cards = dict(sorted(chapter_cards.items()))
        with open(os.path.join(path_output_by_chapter, chapter_file), "w") as fp:
            fp.writelines("\n".join([f"{k} = {v}" for k, v in chapter_cards.items()]))

        chapter_cards = {}
        chapter_id += 1

    if not os.path.exists(marker_complete):
        os.makedirs(path_output_by_chapter, exist_ok=True)
        if path_output_cumulative is not None:
            os.makedirs(path_output_cumulative, exist_ok=True)
        pbar = tqdm(sorted(os.listdir(path_input_cards)))

        for page_file in pbar:
            if not page_file.startswith("page") or not page_file.endswith(".txt"):
                continue
            page_num = int(page_file[4:-4])
            path_in = os.path.join(path_input_cards, page_file)

            with open(path_in, "r") as fp:
                page_cards = fp.read()
            if strip_list is not None:
                for s in strip_list:
                    page_cards = page_cards.replace(s, "")
            page_cards = page_cards.split("\n")
            page_cards = [line.split("=") for line in page_cards]
            assert all([len(splits) == 2 for splits in page_cards])
            page_cards = {splits[0].strip(): splits[1].strip() for splits in page_cards}
            if kill_list is not None:
                page_cards = {k: v for k, v in page_cards.items() if k not in kill_list}
            for k, v in page_cards.items():
                if k not in chapter_cards or len(chapter_cards[k]) < len(v):
                    chapter_cards[k] = v

            if chapter_pages is not None and page_num in chapter_pages:
                dump_chapter()

        if len(chapter_cards) > 0:
            dump_chapter()

        with open(marker_complete, "w") as _:
            pass

    out = [path_output_by_chapter]
    if chapter_pages is not None:
        out.append(path_output_cumulative)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract language cards from a PDF")
    parser.add_argument("pdf_path", type=str, help="PDF input")
    parser.add_argument(
        "--pdf_page_first", type=int, default=None, help="First page to process"
    )
    parser.add_argument(
        "--pdf_page_last", type=int, default=None, help="Last page to process"
    )
    parser.add_argument(
        "--out_path", type=str, default=None, help="Output directory path"
    )
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Optical recognition resolution"
    )
    parser.add_argument(
        "--chapter_pages",
        type=int,
        nargs="+",
        default=None,
        help="List of page numbers with new chapters",
    )
    parser.add_argument("--kill", type=str, nargs="+", default=None, help="Cards keys to remove from the output")
    parser.add_argument("--strip", type=str, nargs="+", default=None, help="Texts to strip from cards")
    parser.add_argument(
        "--verbose", action="store_true", help="Output more details about the process"
    )
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.splitext(args.pdf_path)[0] + "_output"

    openai_api_key = get_openai_api_key()

    out_path_pages_raw = process_pdf_raw(
        args.pdf_path,
        args.out_path,
        first_page=args.pdf_page_first,
        last_page=args.pdf_page_last,
    )
    out_path_pages_resized = resize_pages(args.out_path, longest_side=args.resolution)
    out_path_ocr = ocr_all_pages(args.out_path)
    out_path_cards_all_pages = extract_cards_all_pages(
        args.out_path, verbose=args.verbose
    )

    out_paths_cards_chapters = group_chapters(args.out_path, args.chapter_pages, args.kill, args.strip)

    print("Done.")
