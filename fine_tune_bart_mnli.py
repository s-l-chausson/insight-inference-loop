import time
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BartForSequenceClassification, AdamW, get_linear_schedule_with_warmup


class BART_MNLI_FineTuner:

    def __init__(self, hypotheses, batch_size, nb_classes, nb_epochs, warmup_ratio, device, tokenizer, training_sizes):
        self.hypotheses = hypotheses
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epochs = nb_epochs
        self.warmup_ratio = warmup_ratio
        self.device = device
        self.tokenizer = tokenizer
        self.training_sizes = training_sizes

    def format_nli_trainset(self, df_train=None):
        print(f"Length of df_train before formatting step: {len(df_train)}.")
        length_original_data_train = len(df_train)

        df_train_lst = []
        for label_text, hypothesis in self.hypotheses.items():
            ## entailment
            df_train_step = df_train[df_train.label == label_text].copy(deep=True)
            df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
            df_train_step["all_text_w_hypothesis"] = df_train_step["all_text"].apply(
                lambda x: x + " </s> " + hypothesis)
            df_train_step["label_nli"] = [2] * len(
                df_train_step)  # BART_MNLI returns prob. distribution over the following three classes in the given order: [contradictions, neutral, entailment]
            ## not_entailment
            df_train_step_not_entail = df_train[df_train.label != label_text].copy(deep=True)
            df_train_step_not_entail = df_train_step_not_entail.sample(
                n=min(len(df_train_step), len(df_train_step_not_entail)))
            df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
            df_train_step_not_entail["all_text_w_hypothesis"] = df_train_step_not_entail["all_text"].apply(
                lambda x: x + " </s> " + hypothesis)
            df_train_step_not_entail["label_nli"] = [0] * len(df_train_step_not_entail)
            # append
            df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
        df_train = pd.concat(df_train_lst)

        # shuffle
        df_train = df_train.sample(frac=1)
        df_train["label_nli"] = df_train.label_nli.apply(int)
        df_train["label_nli_explicit"] = ["True" if label == 2 else "Not-True" for label in
                                          df_train["label_nli"]]  # adding this just to simplify readibility

        print(
            f"After adding not_entailment training examples, the training data was augmented to {len(df_train)} texts.")
        print(
            f"Max augmentation could be: len(df_train) * 2 = {length_original_data_train * 2}. It can also be lower, if there are more entail examples than not-entail for a majority class.")

        return df_train

    ## function for reformatting the test set
    def format_nli_testset(self, df_test=None):

        for label, hypothesis in self.hypotheses.items():
            df_test["all_text_w_hypothesis_" + label] = df_test["all_text"].apply(lambda x: x + " </s> " + hypothesis)

        return df_test

    def create_BART_dataset(self, source):
        # source = pd.read_csv(file_name)
        input_ids = []
        attention_masks = []
        labels = []

        for index, row in source.iterrows():
            dic = self.tokenizer.encode_plus(row['all_text_w_hypothesis'],  # Sentence to encode.
                                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                        max_length=512,  # Pad & truncate all sentences.
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',  # Return pytorch tensors.
                                        )
            input_ids.append(dic['input_ids'])
            attention_masks.append(dic['attention_mask'])
            labels.append(row['label_nli'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        return dataset

    def create_BART_test_dataset(self, source, text_field):
        # source = pd.read_csv(file_name)
        input_ids = []
        attention_masks = []
        labels = []

        for index, row in source.iterrows():
            dic = self.tokenizer.encode_plus(row[text_field],  # Sentence to encode.
                                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                        max_length=512,  # Pad & truncate all sentences.
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',  # Return pytorch tensors.
                                        )
            input_ids.append(dic['input_ids'])
            attention_masks.append(dic['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(input_ids, attention_masks)

        return dataset

    def epoch_time(self, start_time, end_time):
        '''Track training time. '''
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, model, iterator, optimizer, scheduler):
        '''Train the model with specified data, optimizer, and loss function. '''
        epoch_loss = 0

        model.train()

        for batch in iterator:
            b_input_ids = batch[0].to(self.device)
            b_attention_masks = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Reset the gradient to not use them in multiple passes
            optimizer.zero_grad()
            loss = model(b_input_ids, b_attention_masks, labels=b_labels, return_dict=True).loss

            # Backprop
            loss.backward()

            # Optimize the weights
            optimizer.step()
            scheduler.step()

            # Record accuracy and loss
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, model, iterator):
        '''Evaluate model performance. '''
        epoch_loss = 0

        # Turm off dropout while evaluating
        model.eval()

        # No need to backprop in eval
        with torch.no_grad():
            for batch in iterator:
                b_input_ids = batch[0].to(self.device)
                b_attention_masks = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                loss = model(b_input_ids, b_attention_masks, labels=b_labels, return_dict=True).loss
                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def train_and_evaluate(self, model, optimizer, scheduler, train_iterator, valid_iterator, path,
                           epochs=10, model_name='model', early_stopping=True):
        best_valid_loss = float('inf')
        val_loss = []
        tr_loss = []

        for epoch in range(epochs):

            # Calculate training time
            start_time = time.time()

            # Get epoch losses and accuracies
            train_loss = self.train(model, train_iterator, optimizer, scheduler)
            valid_loss = self.evaluate(model, valid_iterator)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            # Save training metrics
            val_loss.append(valid_loss)
            tr_loss.append(train_loss)

            # wandb.log({'accuracy': train_acc, 'loss': train_loss})

            print(f'Epoch: {epoch + 1:2} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                # model.save_pretrained(path + "/" + model_name + "_best.pt")
                # Save if best epoch
                torch.save({'epoch': epoch,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'valid_loss': valid_loss}, path + "/" + model_name + "_best.pt")

            # elif valid_loss > best_valid_loss:
            #     break

    def predict(self, model, iterator):
        '''Predict using model. '''
        results = list()

        # Turm off dropout while evaluating
        model.eval()

        # No need to backprop in eval
        with torch.no_grad():
            for batch in tqdm(iterator):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)

                logits = model(b_input_ids,
                               attention_mask=b_input_mask,
                               return_dict=True).logits

                logits = torch.nn.functional.softmax(logits, dim=1)
                results += logits.cpu().detach().tolist()

        return results

    def __call__(self, df_train, df_dev, df_test):
        for T in self.training_sizes:

            print("Using training set of size", T)
            if len(df_train) < T:
                print("NOT ENOUGH DATAPOINTS FOR THIS SPLIT, USING", len(df_train))

            if 'BART_sentiment_pred_' + str(T) in df_test.columns:
                print("SKIPPING THIS SAMPLE SIZE")
                continue

            # Load BartForSequenceClassification, the pretrained BART model with a single linear classification layer on top.
            MODEL = BartForSequenceClassification.from_pretrained(
                "facebook/bart-large-mnli",  # Use the 12-layer BART model, with an uncased vocab.
            )
            MODEL.to(self.device)

            OPTIMIZER = AdamW(MODEL.parameters(), lr=2e-5)

            # Upsample the data
            individual_class_size = T // self.nb_classes
            dfs_list = list()

            for label in ["positive", "negative"]:
                sub = df_train[df_train['label'] == label]
                nb_repeats = individual_class_size // len(sub)
                dfs_list += [sub] * nb_repeats
                nb_remainder = individual_class_size % len(sub)
                dfs_list.append(sub.sample(nb_remainder, replace=True))

            df_TRAIN_UP = pd.concat(dfs_list).sample(frac=1)

            df_TRAIN_UP_FRMTD = self.format_nli_trainset(df_train=df_TRAIN_UP)
            df_dev_FRMTD = self.format_nli_trainset(df_train=df_dev)
            df_test_FRMTD = self.format_nli_testset(df_test=df_test)

            print(df_TRAIN_UP_FRMTD["label"].value_counts())
            print(df_TRAIN_UP_FRMTD["label_nli"].value_counts())

            # Create datasets
            training_set = self.create_BART_dataset(df_TRAIN_UP_FRMTD)
            validation_set = self.create_BART_dataset(df_dev_FRMTD)

            train_iterator = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
            valid_iterator = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)

            total_steps = len(train_iterator) * self.nb_epochs
            warmup_steps = int(self.warmup_ratio * total_steps)

            scheduler = get_linear_schedule_with_warmup(OPTIMIZER,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)

            # Fine-tune model
            self.train_and_evaluate(MODEL, OPTIMIZER, scheduler, train_iterator, valid_iterator, '.',
                               model_name='BART_model_sentiment', early_stopping=False, epochs=self.nb_epochs)

            del OPTIMIZER, train_iterator, valid_iterator

            # Run inference on test set
            checkpoint = torch.load("BART_model_sentiment_best.pt", map_location='cpu')
            print("Epoch:", checkpoint['epoch'])
            MODEL.load_state_dict(checkpoint['model_state'])

            for label in self.hypotheses:
                testing_set = self.create_BART_test_dataset(df_test_FRMTD, "all_text_w_hypothesis_" + label)
                test_iterator = DataLoader(testing_set, batch_size=self.batch_size, shuffle=False)
                df_test_FRMTD['BART_sentiment_pred_' + label + "_" + str(T)] = self.predict(MODEL, test_iterator)
                del test_iterator

            df_test_FRMTD.to_csv('./Economy_models/testing_sentiment_NLI_PROC_' + str(T) + '.csv', index=False)

            if len(df_train) < T:
                print("TERMINATING NOW")
                break

            del MODEL
            torch.cuda.empty_cache()

        print("DONE!")