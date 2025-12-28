import itertools
import time
import argparse
from collections import Counter, defaultdict
from typing import List, Set, Tuple, Optional

from ortools.sat.python import cp_model
from wordfreq import top_n_list

ALPHABET = set("abcdefghijklmnopqrstuvwxyzæøå")

def letters_in_title(s: str) -> Set[str]:
    s = s.lower()
    return {ch for ch in s if ch in ALPHABET}

def build_candidates(titles: List[str]) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    L = [letters_in_title(t) for t in titles]
    by_letter = defaultdict(list)
    for i, ls in enumerate(L):
        for ch in ls:
            by_letter[ch].append(i)

    cands = []
    for ch, idxs in by_letter.items():
        if len(idxs) < 4:
            continue
        for group in itertools.combinations(idxs, 4):
            inter = L[group[0]] & L[group[1]] & L[group[2]] & L[group[3]]
            if inter == {ch}:
                cands.append((ch, group))
    return cands

def danish_words_length(n: int, limit: int = 250000) -> List[str]:
    out = []
    seen = set()
    for w in top_n_list("da", limit):
        w = w.lower()
        if len(w) != n:
            continue
        if any(ch not in ALPHABET for ch in w):
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out

def solve_for_word_anagram(
    titles: List[str],
    candidates: List[Tuple[str, Tuple[int,int,int,int]]],
    k_groups: int,
    word: str,
):
    n = len(titles)
    model = cp_model.CpModel()

    x = [model.NewBoolVar(f"g{i}") for i in range(len(candidates))]

    model.Add(sum(x) == k_groups)

    used_by_song = [[] for _ in range(n)]
    for gi, (_ch, group) in enumerate(candidates):
        for si in group:
            used_by_song[si].append(x[gi])
    for si in range(n):
        if used_by_song[si]:
            model.Add(sum(used_by_song[si]) <= 1)

    cnt = Counter(word)
    all_letters = set(cnt.keys()) | {ch for ch, _ in candidates}
    for ch in all_letters:
        model.Add(
            sum(x[i] for i, (c, _g) in enumerate(candidates) if c == ch) == cnt.get(ch, 0)
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = []
    used = set()
    for i, (ch, group) in enumerate(candidates):
        if solver.Value(x[i]) == 1:
            chosen.append((ch, group))
            used.update(group)

    leftover = [i for i in range(n) if i not in used]
    return chosen, leftover


def load_titles_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f]
    titles = [t for t in titles if t]
    if not titles:
        raise ValueError("Title file is empty (no non-blank lines found).")
    return titles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find disjoint 4-song groups whose shared letters anagram to a Danish word.")
    p.add_argument("titles_file", help="Path to newline-separated song titles.")
    p.add_argument("--min-len", type=int, default=3, help="Minimum word length to try.")
    p.add_argument("--max-len", type=int, default=None, help="Maximum word length to try (defaults to floor(n_titles/4)).")
    p.add_argument("--max-words", type=int, default=3, help="Max matches to print per word length.")
    p.add_argument("--wordfreq-limit", type=int, default=250000, help="How many top Danish words to consider.")
    p.add_argument("--time-limit", type=float, default=15.0, help="CP-SAT time limit per word (seconds).")
    p.add_argument("--workers", type=int, default=8, help="CP-SAT search workers.")
    p.add_argument("--status-every", type=int, default=250, help="Print a status line every N candidate words scanned.")
    return p.parse_args()


def solve_for_word_anagram_with_params(
    titles: List[str],
    candidates: List[Tuple[str, Tuple[int, int, int, int]]],
    k_groups: int,
    word: str,
    time_limit: float,
    workers: int,
):
    n = len(titles)
    model = cp_model.CpModel()

    x = [model.NewBoolVar(f"g{i}") for i in range(len(candidates))]
    model.Add(sum(x) == k_groups)

    used_by_song = [[] for _ in range(n)]
    for gi, (_ch, group) in enumerate(candidates):
        for si in group:
            used_by_song[si].append(x[gi])
    for si in range(n):
        if used_by_song[si]:
            model.Add(sum(used_by_song[si]) <= 1)

    cnt = Counter(word)
    all_letters = set(cnt.keys()) | {ch for ch, _ in candidates}
    for ch in all_letters:
        model.Add(
            sum(x[i] for i, (c, _g) in enumerate(candidates) if c == ch) == cnt.get(ch, 0)
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = []
    used = set()
    for i, (ch, group) in enumerate(candidates):
        if solver.Value(x[i]) == 1:
            chosen.append((ch, group))
            used.update(group)

    leftover = [i for i in range(n) if i not in used]
    return chosen, leftover

def main():
    args = parse_args()
    titles = load_titles_from_file(args.titles_file)
    n = len(titles)
    max_len = args.max_len if args.max_len is not None else (n // 4)
    min_len = max(1, int(args.min_len))

    print(f"Loaded {n} titles from: {args.titles_file}")
    print(f"Word lengths: {min_len}..{max_len}")

    if max_len < min_len:
        print("Nothing to do: max word length is smaller than min word length.")
        return

    candidates = build_candidates(titles)
    print(f"Candidate 4-song groups (shared-letter intersection == 1): {len(candidates)}")

    avail_by_letter = Counter(ch for ch, _ in candidates)
    candidate_letters = set(avail_by_letter.keys())

    print(f"Distinct shared letters among candidates: {len(candidate_letters)}")

    for m in range(min_len, max_len + 1):
        t0 = time.time()
        words = danish_words_length(m, limit=int(args.wordfreq_limit))

        found = 0
        print(f"\n=== Length {m} (max {args.max_words} matches) ===")
        print(f"Words to scan: {len(words)}")

        scanned = 0
        pruned_missing_letter = 0
        pruned_insufficient_supply = 0
        attempted = 0
        for w in words:
            scanned += 1
            cnt = Counter(w)
            if any(ch not in candidate_letters for ch in cnt):
                pruned_missing_letter += 1
                continue
            if any(cnt[ch] > avail_by_letter[ch] for ch in cnt):
                pruned_insufficient_supply += 1
                continue

            attempted += 1
            if args.status_every > 0 and attempted % int(args.status_every) == 0:
                elapsed = time.time() - t0
                rate = attempted / elapsed if elapsed > 0 else 0.0
                print(
                    f"Status: attempted={attempted} scanned={scanned} "
                    f"pruned_missing_letter={pruned_missing_letter} pruned_insufficient_supply={pruned_insufficient_supply} "
                    f"elapsed={elapsed:.1f}s ({rate:.1f} tries/s)"
                )

            res = solve_for_word_anagram_with_params(
                titles,
                candidates,
                k_groups=m,
                word=w,
                time_limit=float(args.time_limit),
                workers=int(args.workers),
            )
            if res is None:
                continue

            chosen, leftover = res
            found += 1

            print(f"\nMATCH WORD: {w}  (uses {4*m} titles, {n-4*m} leftover)")
            for ch, group in sorted(chosen, key=lambda t: t[0]):
                print(f"\nLetter: {ch}")
                for si in group:
                    print(f"  - {titles[si]}")
            print("\nLeftover:", [titles[i] for i in leftover])

            if found >= int(args.max_words):
                break

        if found == 0:
            elapsed = time.time() - t0
            print(
                "No words found for this length. "
                f"(scanned={scanned}, pruned_missing_letter={pruned_missing_letter}, "
                f"pruned_insufficient_supply={pruned_insufficient_supply}, attempted={attempted}, elapsed={elapsed:.1f}s)"
            )
        else:
            elapsed = time.time() - t0
            print(f"Done with length {m}. Found {found} match(es). Elapsed {elapsed:.1f}s")

if __name__ == "__main__":
    main()
