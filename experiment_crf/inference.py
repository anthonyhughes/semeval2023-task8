from typing import Tuple, List

import pycrfsuite
import pandas as pd
from user_args import parse_inference_arguments
from feature_extraction import doc_to_features

import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

args = parse_inference_arguments()


# ====================
def generate_predictions(pd_frame: pd.DataFrame, model: pycrfsuite.Tagger) -> Tuple[List, List]:
    """Generate the predictions for a given a pandas frame"""

    features = doc_to_features(pd_frame)
    words = [row['token'] for (i, row) in pd_frame.iterrows()]
    return words, model.tag(features)


# ====================
def infer_picos(pd_frame: pd.DataFrame, tagger: pycrfsuite.Tagger) -> Tuple[List, List]:
    """Classify tokens with PICO classes"""

    return generate_predictions(pd_frame, tagger)


# ====================
def read_trained_model(trained_model_path: str) -> pycrfsuite.Tagger:
    """Restore a trained model"""
    print('Reading model')
    tagger = pycrfsuite.Tagger()
    tagger.open(trained_model_path)
    return tagger


# ====================
def infer(model: pycrfsuite.Tagger,
          input_file: str,
          output_file: str) -> None:
    print('Beginning Inference')
    input_df = pd.read_csv(input_file)
    all_predictions = infer_picos(input_df, model)

    output_df = pd.DataFrame(columns=['subreddit_id', 'post_id', 'words', 'labels'])
    output_df = pd.concat(
        [
            output_df,
            pd.DataFrame((
                {'subreddit_id': input_df['subreddit_id'], 'post_id': input_df['post_id'], 'words': all_predictions[0], 'labels': all_predictions[1]}
            ))
        ]
    )
    output_df.to_csv(output_file, index=False)

    print('Inference Completed')


if __name__ == '__main__':
    print('Starting')
    model = read_trained_model(args.trained_model_path)
    infer(model, args.input_file, args.output_file)
    print('Done')
