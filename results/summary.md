| Input | Model | Label descriptions | Extraction | Evaluation | P | R | F1 |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| Plain text | `gliner_small-v2.5` | No | — | Strict | 0.77 | 0.87 | 0.82 |
| Plain text | `gliner_small-v2.5` | No | — | Relaxed occurrences | 0.77 | 0.87 | 0.82 |
| Plain text | `gliner_small-v2.5` | No | — | Relaxed unique | 0.78 | 0.89 | 0.83 |
| Plain text | `gliner_small-v2.5` | Yes | — | Strict | 0.98 | 0.28 | 0.43 |
| Plain text | `gliner_small-v2.5` | Yes | — | Relaxed occurrences | 0.98 | 0.28 | 0.43 |
| Plain text | `gliner_small-v2.5` | Yes | — | Relaxed unique | 0.98 | 0.43 | 0.59 |
| Plain text | `gliner_large-v2.5` | No | — | Strict | 1.00 | 0.91 | 0.96 |
| Plain text | `gliner_large-v2.5` | No | — | Relaxed occurrences | 1.00 | 0.91 | 0.96 |
| Plain text | `gliner_large-v2.5` | No | — | Relaxed unique | 1.00 | 0.91 | 0.95 |
| Clean PDF | `gliner_small-v2.5` | No | PyMuPDF | Strict | 0.53 | 0.60 | 0.56 |
| Clean PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed occurrences | 0.77 | 0.87 | 0.82 |
| Clean PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed unique | 0.78 | 0.89 | 0.83 |
| Scanned PDF | `gliner_small-v2.5` | No | PyMuPDF | Strict | 0.00 | 0.00 | 0.00 |
| Scanned PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed occurrences | 0.00 | 0.00 | 0.00 |
| Scanned PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed unique | 0.00 | 0.00 | 0.00 |
| Plain text (validation_small) | LLM chat (no API) | Yes (prompt) | — | Strict | 0.99 | 0.97 | 0.98 |
| Plain text (validation_small) | LLM chat (no API) | Yes (prompt) | — | Relaxed occurrences | 0.99 | 0.97 | 0.98 |
| Plain text (validation_small) | LLM chat (no API) | Yes (prompt) | — | Relaxed unique | 0.98 | 1.00 | 0.99 |
