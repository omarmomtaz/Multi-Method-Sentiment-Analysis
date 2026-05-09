# Multi-Method Sentiment Analysis

A Python-based sentiment analysis tool that compares three different methods: Hugging Face Transformers (DistilBERT), VADER, and TextBlob. Includes interactive mode, review analysis, and visualisation.

## Features

- **Three Analysis Methods**
  - Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` (deep learning)
  - VADER (rule-based, tuned for social media)
  - TextBlob (polarity & subjectivity)
- **Comparison Mode** – Analyse a single text with all three methods and see consensus.
- **Product Review Analyser** – Batch-process multiple reviews with summary statistics.
- **Interactive CLI** – Enter texts on the fly, switch between quick analysis and full comparison.
- **Visualisation** – Generates a horizontal bar chart (`sentiment_comparison.png`) comparing scores across methods for multiple texts.
- **Colour-coded Output** – Terminal output uses green/red for positive/negative sentiments.

## Requirements

- Python 3.8+
- Libraries: `transformers`, `torch`, `vaderSentiment`, `textblob`, `matplotlib`, `numpy`

Install all dependencies with:

```bash
pip install transformers torch vaderSentiment textblob matplotlib numpy
```

**Note:** On first run, Hugging Face will download the DistilBERT model (~260 MB).

## Installation

1. Clone the repository or download `main.py`.
2. Install the requirements (see above).
3. (Optional) Download the `punkt` tokenizer for TextBlob (if not already present):

   ```bash
   python -m textblob.download_corpora
   ```

## Usage

Run the main script:

```bash
python main.py
```

You'll be presented with a menu:

```
Choose mode:
  1. Demo with examples
  2. Compare all three methods
  3. Analyze product reviews
  4. Interactive mode
  5. Quick single text analysis
```

- **Demo** – Runs built‑in examples and creates a visualisation.
- **Compare** – Enter any text and see outputs from all three analysers.
- **Product Reviews** – Analyses a hardcoded set of reviews and shows aggregate sentiment.
- **Interactive** – Repeatedly enter texts for quick Hugging Face analysis, or type `compare` for full comparison.
- **Quick Single** – One‑off Hugging Face analysis.

## Example Output (compare mode)

```
Enter text to compare: I absolutely love this product! It's amazing and works perfectly!

======================================================================
ANALYZING: "I absolutely love this product! It's amazing and works perfectly!"
======================================================================

Results:

1. Hugging Face (DistilBERT)
   Sentiment: POSITIVE
   Confidence: 99.98%

2. VADER
   Sentiment: POSITIVE
   Compound Score: 0.887
   Positive: 0.429
   Neutral: 0.571
   Negative: 0.000

3. TextBlob
   Sentiment: POSITIVE
   Polarity: 0.625 (range: -1 to 1)
   Subjectivity: 0.750 (range: 0 to 1)

----------------------------------------------------------------------
✓ All methods agree: POSITIVE
```

## File Structure

```
.
├── main.py                    # Complete sentiment analysis application
├── sentiment_comparison.png   # Generated visualisation (created at runtime)
└── README.md                  # This file
```

## Customisation

- **Change default model** – Replace the model ID in the `pipeline()` call (line ~13) with any Hugging Face sentiment model.
- **Adjust VADER threshold** – Modify the `compound` thresholds in `analyze_vader()` (currently ±0.05).
- **Add more texts** – Edit the `examples` or `reviews` lists in the `demo_examples()` and `analyze_reviews()` functions.
- **Visualisation** – Tweak `figsize`, colour map, or save path inside `visualize_sentiment_comparison()`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'textblob'`** – Run `pip install textblob`.
- **Hugging Face model download hangs** – Ensure you have a stable internet connection; the model is ~260 MB.
- **VADER `compound` is always 0** – Make sure you installed `vaderSentiment` and import is correct.
- **TextBlob `punkt` missing** – Run `python -m textblob.download_corpora`.

## Contributing

Pull requests, issues, and suggestions are welcome.  
Please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the Transformers library and DistilBERT model.
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) by C.J. Hutto and Eric Gilbert.
- [TextBlob](https://textblob.readthedocs.io/) by Steven Loria.
