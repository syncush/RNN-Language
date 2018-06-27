from __future__ import unicode_literals, print_function, division
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from io import open
import glob
import torch
import random
import unicodedata
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def find_files(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


# loader class for multiple uses
class Loader(object):
    def __init__(self, path):
        self.all_categories = []
        self.category_lines = {}
        for filename in find_files(path):
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = read_lines(filename)
            self.category_lines[category] = lines

    def random_training_example(self):
        category = random_choice(self.all_categories)
        line = random_choice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = line_to_tensor(line)
        return category, line, category_tensor, line_tensor

    def category_from_output(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w_x = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.w_h(hidden) + self.w_x(input))
        output = self.w_o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def train(category_tensor, line_tensor, rnn, criterion, optimizer):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


def evaluate(line_tensor, rnn):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def main():
    letters_size = 57
    hidden_size = 50
    output_size = 18
    learning_rate = 0.005

    rnn = RNN(letters_size, hidden_size, output_size)
    trainLoader = Loader('./data/train/*.txt')
    testLoader = Loader('./data/test/*.txt')

    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    n_epochs = 100000
    print_every = 5000
    plot_every = 1000

  # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

  # training
    for epoch in range(1, n_epochs + 1):
         # Get a random training input and target
        category, line, category_tensor, line_tensor = trainLoader.random_training_example()
        output, loss = train(category_tensor, line_tensor, rnn, criterion, optimizer)
        current_loss += loss

      # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = trainLoader.category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, loss, line, guess, correct))

      # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

  # evaluation
    confusion = torch.zeros(output_size, output_size)
    n_confusion = 10000
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = testLoader.random_training_example()
        output = evaluate(line_tensor, rnn)
        guess, guess_i = testLoader.category_from_output(output)
        category_i = testLoader.all_categories.index(category)
        confusion[category_i][guess_i] += 1

  # Normalize by dividing every row by its sum
    for i in range(output_size):
        confusion[i] = confusion[i] / confusion[i].sum()

  # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

  # Set up axes
    ax.set_xticklabels([''] + testLoader.all_categories, rotation=90)
    ax.set_yticklabels([''] + testLoader.all_categories)

  # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


if __name__ == "__main__":
    main()
