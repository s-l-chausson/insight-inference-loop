import time
import torch
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt


class BERT_FineTuner:

    def __init__(self, label_to_ix, device, tokenizer):
        self.label_to_ix = label_to_ix
        self.device = device
        self.tokenizer = tokenizer

    def create_BERT_training_dataset(self, source):

        # source = pd.read_csv(file_name)
        input_ids = []
        attention_masks = []
        labels = []

        for index, row in source.iterrows():

            dic = self.tokenizer.encode_plus(row['all_text'],                       # Sentence to encode.
                                        add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
                                        max_length = 500,           # Pad & truncate all sentences.
                                        padding = 'max_length',
                                        truncation = True,
                                        return_attention_mask = True,      # Construct attn. masks.
                                        return_tensors = 'pt',             # Return pytorch tensors.
                       )
            input_ids.append(dic['input_ids'])
            attention_masks.append(dic['attention_mask'])
            labels.append(self.label_to_ix[row['label']])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        return dataset


    def create_BERT_testing_dataset(self, source):

        # source = pd.read_csv(file_name)
        input_ids = []
        attention_masks = []
        labels = []

        for index, row in source.iterrows():

            dic = self.tokenizer.encode_plus(row['all_text'],                       # Sentence to encode.
                                        add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
                                        max_length = 500,           # Pad & truncate all sentences.
                                        padding = 'max_length',
                                        truncation = True,
                                        return_attention_mask = True,      # Construct attn. masks.
                                        return_tensors = 'pt',             # Return pytorch tensors.
                       )
            input_ids.append(dic['input_ids'])
            attention_masks.append(dic['attention_mask'])
            labels.append(self.label_to_ix[row['label']])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        return dataset


    def accuracy_binary(self, preds, y):
        """ Return accuracy per batch. """
        argmaxes = torch.argmax(preds, dim=1)
        correct = (argmaxes == y.squeeze(1)).float()
        return correct.sum() / len(correct)


    def epoch_time(self, start_time, end_time):
        '''Track training time. '''
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def show_lossaccevol(self, tr_loss, val_loss, tr_acc, val_acc):
        # Plot accuracy and loss
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].plot(val_loss, label='Validation loss')
        ax[0].plot(tr_loss, label='Training loss')
        ax[0].set_title('Losses')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[1].plot(val_acc, label='Validation accuracy')
        ax[1].plot(tr_acc, label='Training accuracy')
        ax[1].set_title('Accuracies')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        plt.legend()
        plt.show()


    def train(self, model, iterator, optimizer):
        '''Train the model with specified data, optimizer, and loss function. '''
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Reset the gradient to not use them in multiple passes
            optimizer.zero_grad()
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            # print(type(logits))
            # print('Logits:', logits)
            # print('Labels:', b_labels)
            acc = self.accuracy_binary(logits, b_labels.unsqueeze(1))

            # Backprop
            loss.backward()

            # Optimize the weights
            optimizer.step()

            # Record accuracy and loss
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    def evaluate(self, model, iterator):
        '''Evaluate model performance. '''
        epoch_loss = 0
        epoch_acc = 0

        # Turm off dropout while evaluating
        model.eval()

        # No need to backprop in eval
        with torch.no_grad():

            for batch in iterator:

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                loss, logits = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                acc = self.accuracy_binary(logits, b_labels.unsqueeze(1))
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    def train_and_evaluate(self, model, optimizer, train_iterator, valid_iterator, path,
                           epochs = 10, model_name = 'model'):

        best_valid_loss = float('inf')
        best_epoch = 0
        val_loss = []
        val_acc = []
        tr_loss = []
        tr_acc = []

        for epoch in range(epochs):

            # Calculate training time
            start_time = time.time()

            # Get epoch losses and accuracies
            train_loss, train_acc = self.train(model, train_iterator, optimizer)
            valid_loss, valid_acc = self.evaluate(model, valid_iterator)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            # Save training metrics
            val_loss.append(valid_loss)
            val_acc.append(valid_acc)
            tr_loss.append(train_loss)
            tr_acc.append(train_acc)

            # wandb.log({'accuracy': train_acc, 'loss': train_loss})

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch

                #Save every epoch
                torch.save({'epoch': epoch,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'valid_loss': valid_loss}, path + "/" + model_name + "_best.pt")


            print(f'Epoch: {epoch+1:2} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


        # Plot accuracy and loss
        self.show_lossaccevol(tr_loss, val_loss, tr_acc, val_acc)