"""

Description:
This Python script allows users to summarize lengthy articles using
a pre-trained NLP model from the HuggingFace Transformers library.
It supports input via manual text entry or from a .txt file and provides
the option to save the summary results to output files.
"""

import os
import textwrap
from transformers import pipeline

def load_summarizer():
    """
    Loads the pre-trained summarization model.

    Returns:
        summarizer_pipeline (Pipeline): The HuggingFace summarization pipeline.
    """
    model_name = "facebook/bart-large-cnn"
    print(f"\nLoading summarizer model...")
    summarizer_pipeline = pipeline("summarization", model=model_name)
    return summarizer_pipeline


def read_text_input():
    """
    Reads multiline text input from the user.

    Returns:
        full_text (str): Combined text from user input.
    """
    print("\nEnter the text to summarize. Press Enter twice to submit:\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    full_text = " ".join(lines).strip()
    return full_text


def read_from_file():
    """
    Reads input text from a .txt file specified by the user.

    Returns:
        file_content (str): The content read from the file.
    """
    file_path = input("Enter the path to your .txt file: ").strip()
    if not os.path.isfile(file_path):
        print("Error: File not found.")
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def summarize_article(summarizer, article_text, max_length=130, min_length=30):
    """
    Summarizes the given article text using the provided summarizer.

    Args:
        summarizer (Pipeline): The summarization pipeline.
        article_text (str): The input text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        summary_text (str): The summarized output.
    """
    print("\nGenerating summary. Please wait...\n")
    summary = summarizer(
        article_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return summary[0]['summary_text']


def print_wrapped(title, content, width=90):
    """
    Prints a section with a title and word-wrapped content.

    Args:
        title (str): The section title.
        content (str): The text content.
        width (int): The line width for wrapping.
    """
    print(f"\n{title}")
    print("-" * len(title))
    print(textwrap.fill(content, width=width))


def save_to_file(summary_text, original_text):
    """
    Saves the original and summarized text to a text file.

    Args:
        summary_text (str): The generated summary.
        original_text (str): The original input text.
    """
    filename = "summary_output.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ORIGINAL TEXT:\n")
        f.write(original_text + "\n\n")
        f.write("SUMMARY:\n")
        f.write(summary_text)
    print(f"\nSummary saved to: {filename}")


def save_readme_summary(summary_text):
    """
    Saves the summary to a Markdown file.

    Args:
        summary_text (str): The generated summary.
    """
    with open("README_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Summarized Article\n\n")
        f.write(summary_text)
    print("Summary also saved to: README_SUMMARY.md")


def show_summary_stats(original_text, summary_text):
    """
    Displays word count and compression ratio statistics.

    Args:
        original_text (str): The original input text.
        summary_text (str): The generated summary.
    """
    orig_len = len(original_text.split())
    summ_len = len(summary_text.split())
    compression = (summ_len / orig_len) * 100
    print("\nSUMMARY STATISTICS")
    print("-" * 22)
    print(f"Original Length : {orig_len} words")
    print(f"Summary Length  : {summ_len} words")
    print(f"Compression     : {compression:.2f}%")


def main():
    """
    Main function to run the summarization tool.
    """
    print("=" * 80)
    print("TEXT SUMMARIZATION TOOL USING NLP".center(80))
    print("=" * 80)

    summarizer = load_summarizer()

    # Input method selection
    print("\nChoose input method:")
    print("1. Paste text manually")
    print("2. Load from a .txt file")
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        text = read_text_input()
    elif choice == "2":
        text = read_from_file()
    else:
        print("Invalid choice.")
        return

    if not text or len(text.split()) < 40:
        print("\nError: Please provide valid input with at least 40 words.")
        return

    summary = summarize_article(summarizer, text)

    print_wrapped("ORIGINAL TEXT", text)
    print_wrapped("GENERATED SUMMARY", summary)
    show_summary_stats(text, summary)

    # Save to file options
    save_opt = input("\nDo you want to save the summary to files? (y/n): ").strip().lower()
    if save_opt == "y":
        save_to_file(summary, text)
        save_readme_summary(summary)

    print("\nProcess completed!.")


if __name__ == "__main__":
    main()
