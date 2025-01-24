from tqdm import tqdm
import pandas as pd
from transformers import pipeline, logging, AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda import is_available

class NLI_Classifier:

    def __init__(self, traits, model_name, source_column='text', target_column='ZSL'):
        if model_name is None:
            model_name ='facebook/bart-large-mnli'
        use_cuda = is_available()
        if use_cuda:
            print('Using GPU')
            self.classifier = pipeline("zero-shot-classification", model=model_name, device=0)
        else:
            self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.source_column = source_column
        self.target_column = target_column
        self.traits = traits
        if len(self.traits) > 65:
            new_traits = list()
            classes = sorted(list(set([t[:3] for t in self.traits])))
            for c in classes:
                new_traits.append({k:self.traits[k] for k in self.traits if k[:3] == c})
            self.traits = new_traits
        else:
            self.traits = [self.traits]


    def zsl_multi_classifier(self, sequence, context):
        result_json = dict()
        results = self.classifier(sequence, context, hypothesis_template="{}", multi_label=True)
        for label, score in zip(results['labels'], results['scores']):
            result_json[label] = score
        return result_json

    def zsl_multi_classifier_multi_sentences(self, sequences, context, multi_label=True):
        result_json = dict()
        results = self.classifier(sequences, context, hypothesis_template="{}", multi_label=multi_label)
        for r in results:
          result_json[sequence] = dict()
          for sequence, label, score in zip(r['sequence'], r['labels'], r['scores']):
              result_json[sequence][label] = score
        return result_json


    def df_apply_ZSL(self, row):

        results = dict()
        for traits in self.traits:
            results.update(self.zsl_multi_classifier(row[self.source_column], list(traits.values())))

        row[self.target_column] = results
        return row


    def run(self, dataframe, out_file):
        number_lines = len(dataframe)
        chunksize = 12

        if out_file is None:
            out_file_valid = False

        elif isinstance(out_file, str):
            out_file_valid = True

            if os.path.isfile(out_file):
                already_done = pd.read_csv(out_file, names=COLUMNS + [self.target_column])
                start_line = len(already_done)

            else:
                already_done = pd.DataFrame().reindex(columns=dataframe.columns)
                start_line = 0

        else:
            print('ERROR: "out_file" is of the wrong type, expected str')

        for i in tqdm(range(start_line, number_lines, chunksize)):

            sub_df = dataframe.iloc[i: i + chunksize]
            sub_df = sub_df.apply(self.df_apply_ZSL, axis=1)
            already_done = pd.concat([already_done, sub_df])

            if out_file_valid:
                sub_df.to_csv(out_file, mode="a", header=False, index=False)

        return already_done