import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import Shakespeare, one_hot_encoding
from model import CharRNN, CharLSTM
from generate import generate
import warnings
warnings.filterwarnings(action='ignore')

def train(model, trn_loader, device, criterion, optimizer, batch_size, network_type):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    print('Inside Train')
    model.train()

    total_batch = len(trn_loader)
    trn_loss = 0


    hidden = model.init_hidden(batch_size)
    # hidden = hidden.to(device)

    for batch_idx, batch in enumerate(trn_loader) :
        x, label = batch
        print(len(x))
        # input sequence x should be form of one hot vector
        x = one_hot_encoding(x)
        print(len(x))
        x = x.to(device); label = label.to(device)
        if network_type=='RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(model.num_layer, 100, model.hidden_size)
        else :
            hidden = tuple([each.data for each in hidden])
        optimizer.zero_grad()
        if(len(x)==100):
          output, hidden = model.forward(x, hidden)
        #   label_size = label.size(1)
          cost = criterion(output, label.view(10000).long())
          cost.backward(retain_graph=True)
          optimizer.step()
          trn_loss += cost.item()

    trn_loss = round(trn_loss/total_batch, 3)

    return trn_loss


@torch.no_grad()
def validate(model, val_loader, device, criterion, batch_size, network_type='RNN'):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    print('Inside Validate')
    model.eval()

    total_batch = len(val_loader)
    val_loss = 0

    hidden = model.init_hidden(batch_size)
    # hidden = hidden.to(device)

    for batch_idx, batch in enumerate(val_loader) :
        x, label = batch

        # input sequence x should be form of one hot vector
        x = one_hot_encoding(x)
        x = x.to(device); label = label.to(device)
        if network_type=='RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(1, -1, 512)
        else :
            hidden = tuple([each.data for each in hidden])
        if(len(x)==100):
          output, hidden = model.forward(x, hidden)
          cost = criterion(output, label.view(10000).long())
          val_loss += cost.item()

    val_loss = round(val_loss/total_batch, 3)

    return val_loss


def write2File(listFile, name):
    with open(f'./{name}.txt', 'w') as fp:
        for item in listFile:
            fp.write("%s\n" % item)
        print('Done')

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation.
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    input_file_train = open('shakespeare_train.txt', 'r').read()
    input_file_test = open('shakespeare_valid.txt', 'r').read()
    epochs = 10
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = Shakespeare(input_file_train, is_train=True)
    test_dataset = Shakespeare(input_file_test, is_train=False)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    ##################################################################
    #                   RNN model
    ##################################################################
    rnn_model = CharRNN(input_size=len(train_dataset.char2int),
                    hidden_size=512, num_layer=1).to(device)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.01)
    rnn_criterion = torch.nn.CrossEntropyLoss().to(device)

    rnn_trn_loss = []; rnn_val_loss = []

    print('RNN training start')
    for epoch in range(epochs) :
        train_loss = train(rnn_model, train_data, device, rnn_criterion, rnn_optimizer, batch_size, 'RNN')
        test_loss = validate(rnn_model, test_data, device, rnn_criterion, batch_size, 'RNN')
        print('epoch : {}, train loss : {}, validation loss : {}'.format(epoch+1, train_loss, test_loss))

        torch.save({'optimizer': rnn_optimizer.state_dict(),'model': rnn_model.state_dict(),}, f'./rnn/epoch-{epoch}.pth')
        rnn_trn_loss.append(train_loss); rnn_val_loss.append(test_loss)

        rnn_generated_text = generate(rnn_model, input_file_train[:100], 5, 'RNN', train_dataset.char2int, train_dataset.int2char, True)
        rnn_text = open(f'rnn_epoch_{epoch}.txt', 'w')
        rnn_text.write(rnn_generated_text)
        rnn_text.close()

    rnn_generated_text = generate(rnn_model, 'JULIET', 5, 'RNN', train_dataset.char2int, train_dataset.int2char, False)
    rnn_text = open('rnn.txt', 'w')
    rnn_text.write(rnn_generated_text)
    rnn_text.close()



    ##################################################################
    #                   LSTM model
    ##################################################################
    lstm_model = CharLSTM(input_size=len(train_dataset.char2int),
                    hidden_size=512, num_layer=1).to(device)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    lstm_criterion = torch.nn.CrossEntropyLoss().to(device)

    lstm_trn_loss = []; lstm_val_loss = []
    print('\n LSTM training start')

    for epoch in range(epochs) :
        train_loss = train(lstm_model, train_data, device, lstm_criterion, lstm_optimizer, batch_size, 'LSTM')
        test_loss = validate(lstm_model, test_data, device, lstm_criterion, batch_size, 'LSTM')
        print('epoch : {}, train loss : {}, validation loss : {}'.format(epoch+1, train_loss, test_loss))

        torch.save({'optimizer': lstm_optimizer.state_dict(),'model': lstm_model.state_dict(),}, f'./lstm/epoch-{epoch}.pth')
        lstm_trn_loss.append(train_loss); lstm_val_loss.append(test_loss)

        lstm_generated_text = generate(lstm_model, input_file_train[:100], 5, 'LSTM', train_dataset.char2int, train_dataset.int2char, True)
        lstm_text = open(f'lstm_epoch_{epoch}.txt', 'w')
        lstm_text.write(lstm_generated_text)
        lstm_text.close()
        


    lstm_generated_text = generate(lstm_model, 'JULIET', 5, 'LSTM', train_dataset.char2int, train_dataset.int2char, False)
    lstm_text = open('lstm.txt', 'w')
    lstm_text.write(lstm_generated_text)
    lstm_text.close()
    
    write2File(rnn_trn_loss, 'rnn_trn_loss')
    write2File(rnn_val_loss, 'rnn_val_loss')
    write2File(lstm_trn_loss, 'lstm_trn_loss')
    write2File(rnn_val_loss, 'rnn_val_loss')

    draw_result_plot(rnn_trn_loss, rnn_val_loss, lstm_trn_loss, lstm_val_loss)




def draw_result_plot(rnn_trn, rnn_val, lstm_trn, lstm_val) :
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    rnn_epoch = list(range(len(rnn_trn)))
    lstm_epoch = list(range(len(lstm_trn)))

    axes[0, 0].plot(rnn_epoch, rnn_trn)
    axes[0, 0].set_title('RNN model train loss')

    axes[0, 1].plot(rnn_epoch, rnn_val)
    axes[0, 1].set_title('RNN model validation loss')

    axes[1, 0].plot(lstm_epoch, lstm_trn)
    axes[1, 0].set_title('LSTM model train loss')

    axes[1, 1].plot(lstm_epoch, lstm_val)
    axes[1, 1].set_title('LSTM model validation loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('result.png')


if __name__ == '__main__':
     main()
