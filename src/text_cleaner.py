"""Text cleaning and normalization module for the Automated Resume Screening Tool.

Provides robust functions for normalizing whitespace, standardizing bullet characters,
and sanitizing extracted resume text while strictly preserving critical technical
terminology, frameworks, and programming language tokens.
"""

import re
from typing import List, Tuple

# Common resume bullet glyphs
BULLET_CHARS = r"[•▪■●○◦‣⁃∙★]"

# Technical terms with special characters to protect during punctuation removal
# List of (compiled_regex, canonical_restoration_token)
PROTECTED_TERMS: List[Tuple[re.Pattern, str]] = [
    # C++ and C#
    (re.compile(r"\bc\+\+", re.IGNORECASE), "c++"),
    (re.compile(r"\bc#", re.IGNORECASE), "c#"),
    # .NET and variants
    (re.compile(r"(?:^|(?<=\s))\.net\b", re.IGNORECASE), ".net"),
    (re.compile(r"\basp\.net\b", re.IGNORECASE), "asp.net"),
    # Node.js and JavaScript frameworks/runtimes
    (re.compile(r"\bnode\.js\b", re.IGNORECASE), "node.js"),
    (re.compile(r"\bvue\.js\b", re.IGNORECASE), "vue.js"),
    (re.compile(r"\breact\.js\b", re.IGNORECASE), "react.js"),
    (re.compile(r"\bnext\.js\b", re.IGNORECASE), "next.js"),
    (re.compile(r"\bangular\.js\b", re.IGNORECASE), "angular.js"),
    # CI/CD and composite technical terms
    (re.compile(r"\bci/cd\b", re.IGNORECASE), "ci/cd"),
    (re.compile(r"\bhtml/css\b", re.IGNORECASE), "html/css"),
    (re.compile(r"\btcp/ip\b", re.IGNORECASE), "tcp/ip"),
    (re.compile(r"\bpl/sql\b", re.IGNORECASE), "pl/sql"),
    (re.compile(r"\bi/o\b", re.IGNORECASE), "i/o"),
]


def normalize_whitespace(text: str) -> str:
    """Normalize horizontal spaces, tabs, and excessive newlines.

    Collapses multiple spaces and tabs on each line into a single space, strips
    trailing spaces, and reduces 3 or more consecutive newlines to 2, maintaining
    clear paragraph and section boundaries without merging words.

    Args:
        text: Input text string to normalize.

    Returns:
        Cleaned text with normalized whitespace.

    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, but received {type(text).__name__}.")

    if not text.strip():
        return ""

    # Normalize line break variants (CRLF -> LF, CR -> LF)
    result = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs and multiple horizontal whitespace with a single space
    result = re.sub(r"[^\S\n]+", " ", result)

    # Strip whitespace at the start and end of each line
    result = re.sub(r" ?\n ?", "\n", result)

    # Collapse excessive line breaks (3 or more) to at most 2
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Collapse any remaining multiple spaces
    result = re.sub(r" {2,}", " ", result)

    return result.strip()


def normalize_bullets(text: str) -> str:
    """Standardize diverse resume bullet characters into consistent bullet tokens.

    Handles symbols such as •, ▪, ●, ◦, and standalone hyphens without altering
    meaningful hyphens inside compound technical words (e.g. 'scikit-learn',
    'object-oriented').

    Args:
        text: Input text string containing bullet points.

    Returns:
        Text with standardized '- ' bullet prefixes.

    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, but received {type(text).__name__}.")

    if not text.strip():
        return ""

    result = text

    # Standardize bullet glyphs at line starts
    result = re.sub(r"(?m)^\s*" + BULLET_CHARS + r"\s*", "- ", result)

    # Standardize bullet glyphs preceded by whitespace
    result = re.sub(r"(?<=\s)" + BULLET_CHARS + r"\s*", "- ", result)

    # Standardize any remaining standalone bullet glyphs
    result = re.sub(BULLET_CHARS, "- ", result)

    # Standardize hyphen bullets at line starts
    result = re.sub(r"(?m)^\s*-\s*", "- ", result)

    # Standardize hyphens surrounded by horizontal spaces as bullets
    result = re.sub(r"[^\S\n]+-[^\S\n]+", " - ", result)

    # Clean up double horizontal spaces
    result = re.sub(r"[^\S\n]{2,}", " ", result)

    return result.strip()


def clean_text(text: str) -> str:
    """Sanitize and normalize resume text while preserving technical terminology.

    Executes a multi-stage cleaning pipeline:
    1. Validates input type (raises TypeError for non-strings).
    2. Handles empty/whitespace-only input.
    3. Normalizes bullet glyphs.
    4. Converts text to lowercase.
    5. Masks protected technical terms (e.g. C++, C#, .NET, Node.js, CI/CD, HTML/CSS).
    6. Masks decimal numbers (e.g. 3.11, 3.8), percentages (e.g. 5%), and compound words (e.g. scikit-learn).
    7. Strips bullet hyphens, unnecessary punctuation, and multiple dots.
    8. Unmasks all protected technical terms and symbols.
    9. Collapses all whitespace sequences into single spaces.

    Args:
        text: Raw resume or document text string.

    Returns:
        Cleaned, lowercased, and normalized text string.

    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, but received {type(text).__name__}.")

    if not text.strip():
        return ""

    # Step 1: Normalize bullets
    working = normalize_bullets(text)

    # Step 2: Convert to lowercase
    working = working.lower()

    # Step 3: Mask protected technical terms with lowercase alphanumeric tokens
    for idx, (pattern, _) in enumerate(PROTECTED_TERMS):
        token = f"tokenprotected{idx}x"
        working = pattern.sub(token, working)

    # Step 4: Mask decimal numbers (e.g., GPA 3.8, Python 3.11)
    decimal_tokens = {}

    def _mask_decimal(match: re.Match) -> str:
        token = f"tokendecimal{len(decimal_tokens)}x"
        decimal_tokens[token] = match.group(0)
        return token

    working = re.sub(r"\b\d+\.\d+\b", _mask_decimal, working)

    # Step 5: Mask percentage expressions (e.g., 5%, 25%)
    percent_tokens = {}

    def _mask_percent(match: re.Match) -> str:
        token = f"tokenpercent{len(percent_tokens)}x"
        percent_tokens[token] = match.group(0)
        return token

    working = re.sub(r"\b\d+%", _mask_percent, working)

    # Step 6: Mask hyphenated technical / compound words (e.g., scikit-learn, object-oriented, cand-001)
    hyphen_tokens = {}

    def _mask_hyphen(match: re.Match) -> str:
        token = f"tokenhyphen{len(hyphen_tokens)}x"
        hyphen_tokens[token] = match.group(0)
        return token

    working = re.sub(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b", _mask_hyphen, working)

    # Step 7: Remove standalone bullet hyphens and stray hyphens
    working = re.sub(r"(?:^|(?<=\s))-(?=\s|$)", " ", working)

    # Step 8: Remove slashes not part of protected terms (e.g., 'FastAPI/Flask' -> 'fastapi flask')
    working = re.sub(r"/", " ", working)

    # Step 9: Remove punctuation marks and symbols (except colons attached to words, dots in masked tokens)
    working = re.sub(r"[!?,;\"'“”‘’`()\[\]{}~@$%^&*_=|\\]", " ", working)

    # Remove repeated dots (ellipses)
    working = re.sub(r"\.{2,}", " ", working)

    # Remove single dots that are not part of an internal word/number token
    working = re.sub(r"(?<!\w)\.|\.(?!\w)", " ", working)

    # Normalize colons: preserve colons immediately preceded by a word and followed by space or end (e.g. 'skills: ')
    working = re.sub(r"(?<!\w):|:(?!\s|$)", " ", working)
    working = re.sub(r":{2,}", ":", working)

    # Step 10: Restore protected technical terms
    for idx, (_, restoration) in enumerate(PROTECTED_TERMS):
        token = f"tokenprotected{idx}x"
        working = working.replace(token, restoration)

    # Restore decimal numbers
    for token, original in decimal_tokens.items():
        working = working.replace(token, original)

    # Restore percentages
    for token, original in percent_tokens.items():
        working = working.replace(token, original)

    # Restore hyphenated words
    for token, original in hyphen_tokens.items():
        working = working.replace(token, original)

    # Step 11: Normalize all whitespace (including newlines) into single spaces
    working = re.sub(r"\s+", " ", working)

    return working.strip()
