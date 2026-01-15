#!/usr/bin/env python3
"""
High-performance brute-force Morse decoder with unknown spacing.

Features:
- Continuous bitstream
- A–Z only
- Full English dictionary
- Reverse polarity auto-detection
- Trie-based Morse decoding
- Memoization + beam pruning
"""

# -----------------------------
# Configuration
# -----------------------------

BITSTREAM = (
    "1111000101111011010001011101011011011110011111101000101111101101"
)

DICT_PATH = "english_words.txt"

BEAM_WIDTH = 12000
MAX_WORD_LEN = 15
MAX_RESULTS = 20
ENABLE_REVERSE_POLARITY = True


# -----------------------------
# Morse alphabet
# -----------------------------

MORSE = {
    ".-":"A","-...":"B","-.-.":"C","-..":"D",".":"E",
    "..-.":"F","--.":"G","....":"H","..":"I",".---":"J",
    "-.-":"K",".-..":"L","--":"M","-.":"N","---":"O",
    ".--.":"P","--.-":"Q",".-.":"R","...":"S","-":"T",
    "..-":"U","...-":"V",".--":"W","-..-":"X",
    "-.--":"Y","--..":"Z"
}


# -----------------------------
# Build Morse trie (PERF BOOST)
# -----------------------------

def build_morse_trie(morse_map):
    trie = {}
    for code, letter in morse_map.items():
        node = trie
        for s in code:
            node = node.setdefault(s, {})
        node["$"] = letter
    return trie

MORSE_TRIE = build_morse_trie(MORSE)


# -----------------------------
# Language scoring
# -----------------------------

LETTER_FREQ = {
    "E":1.0,"T":0.9,"A":0.85,"O":0.85,"N":0.8,"R":0.8,
    "I":0.75,"S":0.75,"H":0.7,"L":0.7,"D":0.65
}

def score_letter(c):
    return LETTER_FREQ.get(c, 0.25)

def score_word(word, dictionary):
    score = 0.0
    for c in word:
        score += score_letter(c)
    if word in dictionary:
        score += 5.0
    if len(word) > 12:
        score -= 2.0
    return score


# -----------------------------
# Dictionary loader
# -----------------------------

def load_dictionary(path):
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().upper()
            if w.isalpha():
                words.add(w)
    return words


# -----------------------------
# Brute-force decoder (FAST)
# -----------------------------

def brute_force_decode(bits, dictionary):
    results = []

    # state = (bit_pos, current_word, sentence_tuple, score)
    states = [(0, "", (), 0.0)]

    # Memoization: (pos, current_word) -> best score seen
    visited = {}

    n = len(bits)

    while states:
        next_states = []

        for pos, cur_word, sentence, score in states:

            key = (pos, cur_word)
            if key in visited and visited[key] >= score:
                continue
            visited[key] = score

            # Finished bitstream
            if pos == n:
                if cur_word and cur_word in dictionary:
                    final_sentence = sentence + (cur_word,)
                    final_score = score + score_word(cur_word, dictionary)
                    results.append((" ".join(final_sentence), final_score))
                continue

            # Morse trie walk (instead of slicing strings)
            node = MORSE_TRIE
            i = pos
            while i < n and bits[i] in node:
                node = node[bits[i]]
                i += 1
                if "$" in node and len(cur_word) < MAX_WORD_LEN:
                    letter = node["$"]
                    next_states.append((
                        i,
                        cur_word + letter,
                        sentence,
                        score + score_letter(letter)
                    ))

            # Word boundary
            if cur_word and cur_word in dictionary:
                next_states.append((
                    pos,
                    "",
                    sentence + (cur_word,),
                    score + score_word(cur_word, dictionary) - 0.2
                ))

        # Beam pruning
        next_states.sort(key=lambda x: -x[3])
        states = next_states[:BEAM_WIDTH]

    return sorted(results, key=lambda x: -x[1])[:MAX_RESULTS]


# -----------------------------
# Polarity handling
# -----------------------------

def decode_with_polarity(bitstream, dictionary, invert=False):
    if invert:
        bits = bitstream.translate(str.maketrans("10", "01"))
    else:
        bits = bitstream

    bits = bits.replace("1", ".").replace("0", "-")
    results = brute_force_decode(bits, dictionary)
    return results


# -----------------------------
# Main
# -----------------------------

def main():
    print("[*] Loading dictionary...")
    dictionary = load_dictionary(DICT_PATH)
    print(f"    Loaded {len(dictionary):,} words")

    all_results = []

    print("[*] Decoding normal polarity (1=dot, 0=dash)...")
    all_results.extend(
        decode_with_polarity(BITSTREAM, dictionary, invert=False)
    )

    if ENABLE_REVERSE_POLARITY:
        print("[*] Decoding reverse polarity (1=dash, 0=dot)...")
        all_results.extend(
            decode_with_polarity(BITSTREAM, dictionary, invert=True)
        )

    # Deduplicate by sentence, keep best score
    best = {}
    for text, score in all_results:
        if text not in best or best[text] < score:
            best[text] = score

    ranked = sorted(best.items(), key=lambda x: -x[1])[:MAX_RESULTS]

    print("\nTop results:\n")
    for i, (text, score) in enumerate(ranked, 1):
        print(f"{i:2d}. {score:7.2f}  {text}")


if __name__ == "__main__":
    main()
