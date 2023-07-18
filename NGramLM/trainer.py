from .ngram_lm import NGramLM, UnigramLM
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train_file",
                        type=str,
                        help="The directory containing the train file.",
                        required=True)
    parser.add_argument("-n", "--ngram_size",
                        type=int,
                        default=3,
                        help="The size of the the ngrams.",
                        required=False)
    parser.add_argument("-s", "--smooth",
                        type=float,
                        default=1e-3,
                        help="The value for smoothing the probability\
                            distribution of the language model.",
                        required=False)
    parser.add_argument("-p", "--pad_utterances",
                        help="Pad the utterances by adding fake tokens at the\
                            beginning and ending of each utterance.",
                        action='store_true')
    parser.add_argument("-o", "--out_directory",
                        type=str,
                        help="The directory where the model will be stored.",
                        required=True)
    parser.add_argument("-f", "--out_filename",
                        help="The filename for the model.",
                        required=True)
    
    return parser.parse_args()

def main() -> None:
    """This function will train and save the ngram language model."""
    args = parse_args()
    lm = NGramLM(pad_utterances=args.pad_utterances,
                 ngram_size=args.ngram_size,
                 smooth=args.smooth) if args.ngram_size > 1 else UnigramLM(args.smooth)
    lm.estimate(args.train_file)
    lm.save(args.out_directory, args.out_filename)

if __name__ == "__main__" :
    main()