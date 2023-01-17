from typing import Dict, Tuple, List
import pandas as pd
import pycrfsuite
from user_args import parse_train_arguments
from feature_extraction import doc_to_features, doc_to_classes
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

args = parse_train_arguments()


# ====================
def select_x_y_features(data: pd.DataFrame) -> Tuple[List, List]:
    """
    Select the x and y features from a dataframe
    :param data:
    :return:
    """

    print('Features selection')

    # Generate features/tags
    X_train = doc_to_features(data)
    y_train = doc_to_classes(data)
    return X_train, y_train


# ====================
def read_train_data(training_data_location) -> pd.DataFrame:
    """
    Read in the training data
    :param training_data_location:
    :return: data as pd frame
    """
    print('Reading training data')
    return pd.read_csv(training_data_location)


# ====================
def hyperparameter_selection(
        c1: float = 1.0,
        c2: float = 1e-3,
        max_iterations: int = 75,
        possible_transitions: bool = True) -> Dict:
    return {
        'c1': c1,  # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': max_iterations,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': possible_transitions
    }


# ====================
def train(training_data_location, trained_model_path) -> None:
    """
    Main function for training data from a specified path

    :param training_data_location:
    :param trained_model_path:
    :return:
    """
    print('Beginning train')

    data = read_train_data(training_data_location)
    x, y = select_x_y_features(data)

    print('Building trainer')
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.append(x, y)

    trainer.set_params(
        hyperparameter_selection()
    )

    print(trainer.params())

    print('Starting training of sequence model')
    trainer.train(trained_model_path)
    print('Training complete')


if __name__ == '__main__':
    train(args.training_data_location, args.trained_model_path)
