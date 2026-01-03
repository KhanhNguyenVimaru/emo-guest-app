# Gemini Emotion Test Harness

`test_gemini_blocks.py` sends Gemini batches of four sentences from the Hugging Face `emotion` dataset and compares its JSON answers to the gold labels. The steps below reflect exactly how the script works—no `.env` helpers are involved.

## Environment setup
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment under `test/`, then install dependencies:
   ```bash
   cd test
   python -m venv .venv
   .\.venv\Scripts\activate  # PowerShell (use source .venv/bin/activate on macOS/Linux)
   pip install -r requirements.txt
   ```
3. Make your Gemini key available directly in the shell:
   ```powershell
   $env:GEMINI_API_KEY = "AIza..."
   ```
   `VITE_GEMINI_API_KEY` is also recognized, and you can override everything via `--api-key` when running the script.

## Running the tester
Invoke the script from the project root so imports resolve:

```bash
python test/test_gemini_blocks.py --blocks 2 --model gemini-2.5-flash --split test --api-key "<your key>"
```

- `BLOCK_SIZE = 4`, so `--blocks 2` evaluates 8 sentences (2 × 4).
- `--model` defaults to `gemini-2.5-flash`, but any Gemini model string works.
- `--split` chooses which `emotion` subset to read (`test` by default).
- `--api-key` is optional if an environment variable already provides the key.

Under the hood, each block prompt lists four sentences and forces a JSON schema with `block` metadata and a `results` array. The parser strips Markdown fences, normalizes label synonyms (e.g., “happy” → “joy”), and maps every prediction back to its sentence.

## Output
- A `tqdm` progress bar shows how many blocks have been processed.
- The script prints a JSON summary containing accuracy (verified twice), total sentences, and per-sentence diagnostics (`sentence`, `gold_emotion`, `predicted_emotion`, `match`).
- Any malformed JSON response or missing sentence triggers an error so you can adjust prompts or reduce block counts.

## Tips
- The first run downloads the `emotion` dataset automatically through `datasets.load_dataset`.
- Keep `--blocks` small while iterating on prompts to conserve quota; scale up for wider evaluations.
- Because `.env` has been removed, remember to export the API key manually in each new shell session or store it in your preferred secrets manager.
