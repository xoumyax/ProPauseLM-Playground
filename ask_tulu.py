import requests
import argparse
import json
import math
import re
from collections import Counter


# ---------- Lightweight TF-IDF cosine similarity (stdlib only) ----------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Small English stopword list. Removing these makes similarity focus on
# content words, which matters more for "are these answers about the same
# thing" than for grammatical structure.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "of", "as", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "them", "his", "her", "their", "our", "your", "my", "me", "us", "so",
    "not", "no", "nor", "can", "will", "just", "than", "too", "very",
    "s", "t", "d", "ll", "m", "o", "re", "ve", "y",
}


def _tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _tfidf_cosine(text_a, text_b):
    """
    TF-IDF cosine similarity between two strings, computed with only the
    two documents as the corpus. Values in [0, 1]; higher = more similar.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    vocab = set(tf_a) | set(tf_b)

    def df(term):
        return (1 if term in tf_a else 0) + (1 if term in tf_b else 0)

    n_docs = 2
    # Smoothed idf: log((N + 1) / (df + 1)) + 1  (sklearn-style)
    idf = {t: math.log((n_docs + 1) / (df(t) + 1)) + 1 for t in vocab}

    def tfidf_vec(tf):
        return {t: tf.get(t, 0) * idf[t] for t in vocab}

    vec_a = tfidf_vec(tf_a)
    vec_b = tfidf_vec(tf_b)

    dot = sum(vec_a[t] * vec_b[t] for t in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------- DR Tulu query ----------

def _query_tulu(question, endpoint, timeout=1800):
    """
    Send a single question to DR Tulu and return (final_text, token_count).
    Handles several common response formats; falls back to dumping raw JSON.
    """
    response = requests.post(
        endpoint,
        json={"messages": [{"role": "user", "content": question}]},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    # --- EXTRACTION LOGIC ---
    final_text = "No response content found"

    # 1. Standard OpenAI Chat format
    if "choices" in data and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if "message" in choice:
            final_text = choice["message"].get("content", "")
        elif "text" in choice:
            final_text = choice.get("text", "")

    # 2. Ollama format
    elif "response" in data:
        final_text = data["response"]

    # 3. Hugging Face TGI format
    elif isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        final_text = data[0]["generated_text"]

    # 4. Fallback: dump raw JSON
    if final_text == "No response content found" or not final_text:
        print(f"Warning: Unexpected JSON structure for '{question}'. Dumping raw data.")
        final_text = f"RAW JSON: {json.dumps(data)}"

    # --- TOKEN COUNT EXTRACTION ---
    token_count = data.get("tokens")
    if token_count is None:
        print("warning, cannot get token count")
        token_count = int(len(final_text.split()) * 1.3)

    return final_text, token_count


# ---------- Main driver ----------

def ask_dr_tulu_to_json(
    question_pairs,
    output_filename="tulu_results.json",
    base_url="http://localhost:8080",
    similarity_threshold=0.15,
):
    """
    For each (question, question_specific) pair, queries DR Tulu twice
    and compares the two answers using TF-IDF cosine similarity.

    If similarity >= similarity_threshold -> 'clear':
      -> "query" = original question
      -> "final_answer" = answer to the original question

    If similarity < similarity_threshold -> 'vague':
      -> "query" = original question
      -> "clarified_query" = question_specific
      -> "final_answer" = answer to the specific question

    "number of tokens" is the sum across both calls.

    NOTE on threshold: TF-IDF cosine on a 2-document corpus runs lower
    than you might expect. 0.15 is a reasonable starting point for
    paragraph-length answers (similar answers land ~0.2, divergent ~0.04
    in quick tests); tune on a handful of labeled pairs from your data.
    """
    endpoint = f"{base_url}/chat"
    formatted_results = []

    for q, q_specific in question_pairs:
        print(f"Researching:\n  original: '{q}'\n  specific: '{q_specific}'")
        try:
            ans_original, tokens_original = _query_tulu(q, endpoint)

            # If the two questions are identical, skip the second call —
            # the answer would (modulo sampling noise) be the same, so it's
            # definitionally 'clear' and we save a round trip.
            if q.strip() == q_specific.strip():
                print("  questions identical -> skipping second query")
                record = {
                    "query": q,
                    "final_answer": ans_original,
                    "type": "clear",
                    "number of tokens": tokens_original or 0,
                    "similarity": 1.0,
                }
                formatted_results.append(record)
                print(f"  -> classified as '{record['type']}'\n")
                continue

            ans_specific, tokens_specific = _query_tulu(q_specific, endpoint)

            similarity = _tfidf_cosine(ans_original, ans_specific)
            print(f"  similarity = {similarity:.3f} (threshold = {similarity_threshold})")

            total_tokens = (tokens_original or 0) + (tokens_specific or 0)

            if similarity >= similarity_threshold:
                record = {
                    "query": q,
                    "final_answer": ans_original,
                    "type": "clear",
                    "number of tokens": tokens_original,
                    "similarity": round(similarity, 4),
                }
            else:
                record = {
                    "query": q,
                    "clarified_query": q_specific,
                    "final_answer": ans_specific,
                    "type": "vague",
                    "number of tokens": tokens_specific,
                    "similarity": round(similarity, 4),
                }

            formatted_results.append(record)
            print(f"  -> classified as '{record['type']}'\n")

        except requests.exceptions.RequestException as e:
            print(f"Error querying DR Tulu for pair ('{q}', '{q_specific}'): {e}\n")
            formatted_results.append({
                "query": q,
                "clarified_query": q_specific,
                "final_answer": f"Error: {str(e)}",
                "type": "error",
                "number of tokens": 0,
            })

    with open(output_filename, "w", encoding="utf-8") as file:
        json.dump(formatted_results, file, indent=4, ensure_ascii=False)

    print(f"Process complete! All answers saved to '{output_filename}'.")
    return formatted_results


# --- HOW TO RUN IT ---

if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process simplified and clarified questions.")

    # Required positional arguments
    parser.add_argument("simplified", help="Path to the simplified questions file")
    parser.add_argument("clarified", help="Path to the clarified questions file")
    parser.add_argument("output", help="Path to the output JSON file")

    # Optional positional argument
    # nargs='?' makes it optional. You can set a default value and type.
    parser.add_argument("threshold", nargs="?", type=float, default=0.2,
                        help="Optional threshold value (default: 0.2)")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your program
    print(f"Simplified File: {args.simplified}")
    print(f"Clarified File:  {args.clarified}")
    print(f"Output File:     {args.output}")
    print(f"Threshold:       {args.threshold}")

    with open(args.simplified) as f:
        original_questions = f.read().splitlines()
    with open(args.clarified) as f:
        specific_questions = f.read().splitlines()

    if len(original_questions) != len(specific_questions):
        print(
            f"Warning: row count mismatch "
            f"(question={len(original_questions)}, "
            f"question_specific={len(specific_questions)}). "
            "Truncating to the shorter of the two."
        )

    question_pairs = list(zip(original_questions, specific_questions))
    question_pairs = [(a, b) for a, b in question_pairs if a.strip() and b.strip()]

    res = ask_dr_tulu_to_json(
        question_pairs,
        output_filename=args.output,
        base_url="http://localhost:8080",
        similarity_threshold=args.threshold,
    )
