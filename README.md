# Musiz Quiz Generator

A generator for a music quiz where participants have to:

1. guess the title of a song
2. find the shared letter between song titles in each song group
3. find the shared word that the found letters form

---

This problem is a variant of the Exact Cover problem and has been solved with `ortools`.

Install dependencies with `uv sync`, then run the program with `uv run main.py`.

```sh
usage: main.py  [-h] [--lang LANG] [--min-len MIN_LEN] [--max-len MAX_LEN]
                [--max-words MAX_WORDS] [--wordfreq-limit WORDFREQ_LIMIT]
                [--time-limit TIME_LIMIT] [--workers WORKERS][--status-every STATUS_EVERY]
               newline_input.txt
```
