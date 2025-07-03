import argparse
import os
import logging
import pandas as pd
from data_loader import load_and_clean_data
from feature_engineering import add_features
from sentiment_analysis import add_sentiment_scores
from modeling import train_and_evaluate_models
from topic_modeling import lda_topic_modeling, generate_wordcloud
from utils import ensure_dir
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis Pipeline for Data Breaches")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the CSV file with tweets')
    parser.add_argument('--company', type=str, required=True, help='Company name (Chegg, Marriott, UnderArmour, T-mobile)')
    parser.add_argument('--do_sentiment', action='store_true', help='Run sentiment analysis')
    parser.add_argument('--do_modeling', action='store_true', help='Run supervised modeling')
    parser.add_argument('--do_topic_modeling', action='store_true', help='Run topic modeling (LDA)')
    parser.add_argument('--do_wordcloud', action='store_true', help='Generate word cloud')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    logging.info(f"Loading and cleaning data from {args.dataset}")
    df = load_and_clean_data(args.dataset)

    logging.info("Adding engineered features")
    df = add_features(df)

    if args.do_sentiment or args.do_modeling or args.do_topic_modeling or args.do_wordcloud:
        logging.info("Adding sentiment scores")
        df = add_sentiment_scores(df)

    # Save cleaned and feature-engineered data
    cleaned_path = os.path.join(args.output_dir, f"{args.company}_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    logging.info(f"Saved cleaned data to {cleaned_path}")

    if args.do_modeling:
        logging.info("Running supervised modeling")
        # Example: Predict if tweet mentions a breached company (binary target)
        breached_companies = ["Marriott", "Chegg", "UnderArmour", "T-mobile"]
        df["Breached"] = df["text"].apply(lambda x: int(any(c.lower() in x.lower() for c in breached_companies)))
        features = ["Sentiment", "NumWords", "NumHashtags", "NumMentions"]
        X = df[features].values
        y = df["Breached"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        results = train_and_evaluate_models(X_scaled, y)
        results_path = os.path.join(args.output_dir, f"{args.company}_model_results.csv")
        results.to_csv(results_path, index=False)
        logging.info(f"Saved modeling results to {results_path}")
        print(results)

    if args.do_topic_modeling:
        logging.info("Running topic modeling (LDA)")
        lda_model, dictionary, corpus = lda_topic_modeling(df)
        # Optionally, save topics to a file
        topics_path = os.path.join(args.output_dir, f"{args.company}_lda_topics.txt")
        with open(topics_path, 'w') as f:
            for topic in lda_model.show_topics(num_topics=10):
                f.write(str(topic) + '\n')
        logging.info(f"Saved LDA topics to {topics_path}")

    if args.do_wordcloud:
        logging.info("Generating word cloud")
        wordcloud_path = os.path.join(args.output_dir, f"{args.company}_wordcloud.png")
        generate_wordcloud(df, output_path=wordcloud_path)
        logging.info(f"Saved word cloud to {wordcloud_path}")

if __name__ == "__main__":
    main() 