import argparse
from src.predictor import get_signal

def main():
    """
    Main function to run the CLI application.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="AI-based stock prediction application."
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="The stock ticker symbol to predict (e.g., 'RELIANCE.NS')."
    )
    args = parser.parse_args()

    # --- Get and Display Signal ---
    print(f"Generating signal for {args.ticker}...")
    signal = get_signal(args.ticker)
    print(f"\n---------------------------------")
    print(f"Ticker: {args.ticker}")
    print(f"Predicted Signal: {signal}")
    print(f"---------------------------------")
    print("\nDisclaimer: This is a proof-of-concept project. Do not use for actual trading.")

if __name__ == '__main__':
    main()
