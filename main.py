# pip install vaderSentiment

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np

class SentimentAnalyzer:
    """Sentiment analyzer with multiple methods"""
    
    def __init__(self):
        print("Initializing sentiment analyzers...")
        
        # 1. Hugging Face Transformer (DistilBERT)
        print("  Loading Hugging Face model...")
        self.hf_sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # 2. VADER (Valence Aware Dictionary for Sentiment Reasoning)
        print("  Loading VADER...")
        self.vader = SentimentIntensityAnalyzer()
        
        # 3. TextBlob
        print("  TextBlob ready...")
        
        print("✓ All analyzers loaded!\n")
    
    def analyze_huggingface(self, text):
        """Analyze using Hugging Face Transformers"""
        result = self.hf_sentiment(text)[0]
        return {
            'method': 'Hugging Face (DistilBERT)',
            'label': result['label'],
            'score': result['score'],
            'sentiment': result['label'].lower()
        }
    
    def analyze_vader(self, text):
        """Analyze using VADER"""
        scores = self.vader.polarity_scores(text)
        
        # Determine sentiment based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'POSITIVE'
        elif compound <= -0.05:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'method': 'VADER',
            'label': sentiment,
            'score': abs(compound),
            'sentiment': sentiment.lower(),
            'detailed_scores': scores
        }
    
    def analyze_textblob(self, text):
        """Analyze using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'POSITIVE'
        elif polarity < -0.1:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'method': 'TextBlob',
            'label': sentiment,
            'score': abs(polarity),
            'sentiment': sentiment.lower(),
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def analyze_all(self, text):
        """Analyze using all three methods"""
        results = {
            'text': text,
            'huggingface': self.analyze_huggingface(text),
            'vader': self.analyze_vader(text),
            'textblob': self.analyze_textblob(text)
        }
        return results
    
    def compare_methods(self, text):
        """Compare results from all three methods"""
        print("=" * 70)
        print(f"ANALYZING: \"{text}\"")
        print("=" * 70)
        
        results = self.analyze_all(text)
        
        # Display results
        print("\nResults:\n")
        
        # Hugging Face
        hf = results['huggingface']
        print(f"1. {hf['method']}")
        print(f"   Sentiment: {hf['label']}")
        print(f"   Confidence: {hf['score']:.2%}")
        print()
        
        # VADER
        vader = results['vader']
        print(f"2. {vader['method']}")
        print(f"   Sentiment: {vader['label']}")
        print(f"   Compound Score: {vader['detailed_scores']['compound']:.3f}")
        print(f"   Positive: {vader['detailed_scores']['pos']:.3f}")
        print(f"   Neutral: {vader['detailed_scores']['neu']:.3f}")
        print(f"   Negative: {vader['detailed_scores']['neg']:.3f}")
        print()
        
        # TextBlob
        tb = results['textblob']
        print(f"3. {tb['method']}")
        print(f"   Sentiment: {tb['label']}")
        print(f"   Polarity: {tb['polarity']:.3f} (range: -1 to 1)")
        print(f"   Subjectivity: {tb['subjectivity']:.3f} (range: 0 to 1)")
        print()
        
        print("-" * 70)
        
        # Consensus
        sentiments = [hf['sentiment'], vader['sentiment'], tb['sentiment']]
        if len(set(sentiments)) == 1:
            print(f"✓ All methods agree: {sentiments[0].upper()}")
        else:
            from collections import Counter
            most_common = Counter(sentiments).most_common(1)[0][0]
            print(f"⚠ Mixed results. Majority: {most_common.upper()}")
        
        return results


def visualize_sentiment_comparison(texts, analyzer):
    """Visualize sentiment analysis for multiple texts"""
    
    print("\n" + "=" * 70)
    print("ANALYZING MULTIPLE TEXTS")
    print("=" * 70)
    
    results = []
    for text in texts:
        result = analyzer.analyze_all(text)
        results.append(result)
        print(f"\n✓ Analyzed: \"{text[:50]}...\"" if len(text) > 50 else f"\n✓ Analyzed: \"{text}\"")
    
    # Prepare data for visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['huggingface', 'vader', 'textblob']
    method_names = ['Hugging Face', 'VADER', 'TextBlob']
    
    for idx, (method, name) in enumerate(zip(methods, method_names)):
        ax = axes[idx]
        
        # Extract sentiments and scores
        sentiments = []
        scores = []
        colors = []
        
        for result in results:
            sentiment = result[method]['sentiment']
            score = result[method]['score']
            
            sentiments.append(sentiment)
            scores.append(score)
            
            # Color coding
            if sentiment == 'positive':
                colors.append('green')
            elif sentiment == 'negative':
                colors.append('red')
            else:
                colors.append('gray')
        
        # Create bar chart
        y_pos = np.arange(len(texts))
        ax.barh(y_pos, scores, color=colors, alpha=0.7)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Text {i+1}" for i in range(len(texts))], fontsize=9)
        ax.set_xlabel('Confidence/Score', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add sentiment labels
        for i, (score, sentiment) in enumerate(zip(scores, sentiments)):
            ax.text(score + 0.02, i, sentiment.upper(), 
                   va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sentiment_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to 'sentiment_comparison.png'")
    plt.close()


def analyze_reviews(reviews):
    """Analyze product reviews or feedback"""
    
    print("\n" + "=" * 70)
    print("PRODUCT REVIEW SENTIMENT ANALYSIS")
    print("=" * 70)
    
    analyzer = SentimentAnalyzer()
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    print("\nAnalyzing reviews...\n")
    
    for i, review in enumerate(reviews, 1):
        result = analyzer.analyze_huggingface(review)
        sentiment = result['sentiment']
        score = result['score']
        
        if sentiment == 'positive':
            positive_count += 1
            emoji = "😊"
        elif sentiment == 'negative':
            negative_count += 1
            emoji = "😞"
        else:
            neutral_count += 1
            emoji = "😐"
        
        print(f"Review {i}: {emoji} {sentiment.upper()} ({score:.2%})")
        print(f"  \"{review[:60]}...\"" if len(review) > 60 else f"  \"{review}\"")
        print()
    
    # Summary
    total = len(reviews)
    print("-" * 70)
    print("SUMMARY:")
    print(f"  Positive: {positive_count}/{total} ({positive_count/total:.1%})")
    print(f"  Negative: {negative_count}/{total} ({negative_count/total:.1%})")
    print(f"  Neutral: {neutral_count}/{total} ({neutral_count/total:.1%})")
    print("-" * 70)
    
    # Overall sentiment
    if positive_count > negative_count:
        print("\n✓ Overall: POSITIVE sentiment 👍")
    elif negative_count > positive_count:
        print("\n✗ Overall: NEGATIVE sentiment 👎")
    else:
        print("\n⚖ Overall: MIXED/NEUTRAL sentiment")


def interactive_mode(analyzer):
    """Interactive sentiment analysis"""
    
    print("\n" + "=" * 70)
    print("INTERACTIVE SENTIMENT ANALYZER")
    print("=" * 70)
    print("\nEnter text to analyze (or 'quit' to exit)")
    print("Commands:")
    print("  - Type any text for analysis")
    print("  - Type 'compare' to compare all three methods")
    print("  - Type 'quit' to exit")
    print("-" * 70)
    
    while True:
        text = input("\nYour text: ").strip()
        
        if text.lower() == 'quit':
            print("\nGoodbye! 👋")
            break
        
        if not text:
            continue
        
        if text.lower() == 'compare':
            text = input("Enter text to compare: ").strip()
            if text:
                analyzer.compare_methods(text)
        else:
            # Quick analysis with Hugging Face
            result = analyzer.analyze_huggingface(text)
            
            if result['sentiment'] == 'positive':
                emoji = "😊"
                color = "\033[92m"  # Green
            else:
                emoji = "😞"
                color = "\033[91m"  # Red
            
            print(f"\n{color}Sentiment: {result['label']} {emoji}\033[0m")
            print(f"Confidence: {result['score']:.2%}")


def demo_examples():
    """Demonstrate with example texts"""
    
    print("\n" + "=" * 70)
    print("DEMO: SENTIMENT ANALYSIS EXAMPLES")
    print("=" * 70)
    
    examples = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Completely disappointed.",
        "The product is okay. Nothing special, but it works.",
        "Best purchase ever! Highly recommend to everyone!",
        "Terrible quality. Waste of money. Do not buy!",
        "It's fine, I guess. Does what it's supposed to do.",
        "Outstanding service! The team went above and beyond!",
        "Horrible customer support. They never responded to my emails."
    ]
    
    analyzer = SentimentAnalyzer()
    
    print("\nAnalyzing example texts...\n")
    
    for i, text in enumerate(examples, 1):
        result = analyzer.analyze_huggingface(text)
        sentiment = result['sentiment']
        score = result['score']
        
        if sentiment == 'positive':
            emoji = "😊"
            color = "\033[92m"
        else:
            emoji = "😞"
            color = "\033[91m"
        
        print(f"{i}. \"{text}\"")
        print(f"   {color}{sentiment.upper()} {emoji} ({score:.2%})\033[0m\n")
    
    # Visualize comparison
    print("Creating visualization...")
    visualize_sentiment_comparison(examples[:5], analyzer)
    
    return analyzer


def main():
    """Main function"""
    
    print("\nChoose mode:")
    print("  1. Demo with examples")
    print("  2. Compare all three methods")
    print("  3. Analyze product reviews")
    print("  4. Interactive mode")
    print("  5. Quick single text analysis")
    
    choice = input("\nEnter choice (1-5) or press Enter for demo: ").strip()
    
    if choice == "1" or choice == "":
        analyzer = demo_examples()
        
    elif choice == "2":
        analyzer = SentimentAnalyzer()
        text = input("\nEnter text to analyze: ").strip()
        if text:
            analyzer.compare_methods(text)
    
    elif choice == "3":
        reviews = [
            "Great product! Exceeded my expectations.",
            "Arrived broken. Very disappointed with the quality.",
            "It's okay for the price. Nothing spectacular.",
            "Love it! Best purchase I've made this year!",
            "Poor quality. Stopped working after a week.",
            "Exactly as described. Happy with my purchase."
        ]
        analyze_reviews(reviews)
    
    elif choice == "4":
        analyzer = SentimentAnalyzer()
        interactive_mode(analyzer)
    
    elif choice == "5":
        analyzer = SentimentAnalyzer()
        text = input("\nEnter text: ").strip()
        if text:
            result = analyzer.analyze_huggingface(text)
            print(f"\nSentiment: {result['label']}")
            print(f"Confidence: {result['score']:.2%}")
    
    print("\n" + "=" * 70)
    print("✓ Sentiment analysis complete!")
    print("=" * 70)
    print("\nQuick usage:")
    print("  from transformers import pipeline")
    print("  sentiment = pipeline('sentiment-analysis')")
    print("  result = sentiment('I love this!')[0]")
    print("  print(result)")


if __name__ == "__main__":
    main()