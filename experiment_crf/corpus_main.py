from corpus_tools.corpus_utils import generate_all_word_tokens_spans, generate_corpus, \
    update_with_distant_annotations
import pandas as pd


def is_bad_token(token: str) -> bool:
    return token == "" or token.startswith("http")


def create_token_classification_corpus_as_csv(file_location: str,
                                              output_location: str,
                                              target_column: str = 'stage1_labels',
                                              task: str = 'b') -> None:
    """"
        Create a new CSV with 1 to 1 token and labels + additional features
    """
    all_entries_df = pd.read_csv(file_location)
    word_token_spans = generate_all_word_tokens_spans(all_entries_df, target_column)
    final_corpus = generate_corpus(word_token_spans, task)
    if task == 'b':
        final_corpus = update_with_distant_annotations(final_corpus)
    final_frame = pd.DataFrame(final_corpus,
                               columns=[
                                   'token', "clean_token", 'label', 'pos_tag', 'subreddit_id', 'post_id',
                                   "isalnum", "isupper", "isnumeric", "istitle", "has_question_mark", "start_of_doc",
                                   "end_of_doc"
                               ])
    final_frame.to_csv(path_or_buf=output_location, index=False)


def generate_task_b_corpora():
    print('Generating PICO train corpus (Task B)')
    create_token_classification_corpus_as_csv(
        file_location='./medical-corpus/st2/st2_train_inc_text_.csv',
        output_location='./experiment_crf/st2_train_all_cats.csv',
        target_column='stage2_labels'
    )
    print('Generating PICO test corpus ')
    create_token_classification_corpus_as_csv(
        file_location='./medical-corpus/st2/st2_test_inc_text_.csv',
        output_location='./experiment_crf/st2_test_all_cats.csv',
        target_column='stage2_labels'
    )


def generate_task_a_corpora():
    print('Generating claim corpus (Task A)')
    create_token_classification_corpus_as_csv(
        file_location='./medical-corpus/st1/st1_train_inc_text_.csv',
        output_location='./experiment_crf/st1_train_all_cats.csv',
        target_column='stage1_labels',
        task='a'
    )
    print('Generating PICO test corpus ')
    create_token_classification_corpus_as_csv(
        file_location='./medical-corpus/st1/st1_test_inc_text_.csv',
        output_location='./experiment_crf/st1_test_all_cats.csv',
        target_column='stage1_labels',
        task='a'
    )


if __name__ == '__main__':
    generate_task_a_corpora()
