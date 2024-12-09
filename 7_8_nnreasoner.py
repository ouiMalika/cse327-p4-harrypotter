import torch
from torch import Tensor, nn
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import argparse

from kbencoder import (
    generate_autoencoder_embeddings,
    generate_chainbased_embeddings,
    generate_termwalk_embeddings,
    generate_unification_embeddings,
)
from kbparser import parse_atom, parse_rule
from vocab import Vocabulary
from torch.utils.data import random_split, DataLoader, Dataset

hidden_size1 = 30
hidden_size2 = 15
num_classes = 1
batch_size = 25
learning_rate = 0.05
LOSS_STEP = 50

class ReasonerData(Dataset):
    def __init__(self, data, device="cpu") -> None:
        super().__init__()
        self.data = (
            torch.index_select(
                data, 1, torch.tensor(range(data.shape[1] - 1)).to(device)
            )
            .cpu()
            .to_dense()
            .numpy()
        )
        self.labels = (
            torch.index_select(data, 1, torch.tensor([data.shape[1] - 1]).to(device))
            .cpu()
            .to_dense()
            .numpy()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).float().reshape(-1)
        return sample, label

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # No sigmoid here since BCEWithLogitsLoss is used
        return x

def train_reasoning_model(
    training_file, num_epochs, save_file, vocab: Vocabulary,
    embed_type="unification", embed_path="rKB_model.pth", embed_size=50
):
    print("Training " + embed_type)
    print(f"Embed size: {embed_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading examples from " + training_file)
    examples = []
    with open(training_file, mode="r") as f:
        lines = f.readlines()[1:-1]
        lines = [x.lstrip() for x in lines if x.strip() and x[0] != "%"]
        print("Parsing examples...")
        for line in lines:
            goal, rule, score = map(str.strip, line.split("\t"))
            examples.append([parse_atom(goal), parse_rule(rule), float(score)])

    print("Generating embeddings...")
    embeddings_func: dict = {
        "unification": lambda ex, device: generate_unification_embeddings(
            ex, device, vocab, embed_size, embed_path
        ),
        "autoencoder": lambda ex, device: generate_autoencoder_embeddings(
            ex, device, vocab
        ),
        "chainbased": lambda ex, device: generate_chainbased_embeddings(
            ex, device, embed_size
        ),
        "termwalk": lambda ex, device: generate_termwalk_embeddings(
            ex, device, vocab
        ),
    }

    embeddings = embeddings_func.get(embed_type, "unification")(examples, device.type)
    print("Embeddings shape:", embeddings.shape)

    data = ReasonerData(embeddings, device.type)
    data = random_split(data, [int(len(data)*0.9), len(data) - int(len(data)*0.9)])
    train_loader = DataLoader(dataset=data[0], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data[1], batch_size=batch_size, shuffle=True)

    print("Loaded data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Determine input dimension dynamically
    sample, _ = data[0][0]  # get first sample from training split
    input_dim = sample.shape[0]
    print("Actual input dimension to model:", input_dim)

    criterion = torch.nn.BCEWithLogitsLoss()
    model = NeuralNet(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_list = []
    epoch = 0

    primary_smoothing_window = 50
    secondary_smoothing_window = 5
    current_gradient = "-"

    while True:
        running_loss = []
        model.train()
        for (sample, label) in train_loader:
            sample = sample.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            s_out = model(sample)
            loss = criterion(s_out, label)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        loss_list.append(np.mean(running_loss))

        if len(loss_list) > primary_smoothing_window:
            smoothed_data = savgol_filter(loss_list, primary_smoothing_window, 3)
            current_gradient = np.mean(np.gradient(smoothed_data)[-1::-secondary_smoothing_window])
        else:
            current_gradient = "-"

        print(
            f"\r{epoch}\t{str(loss_list[-1])[:7]} ({str(current_gradient)[:7] if current_gradient != '-' else '-'})",
            end=""
        )
        if epoch % LOSS_STEP == 0:
            print()

        epoch += 1

        if epoch >= 750 and epoch % LOSS_STEP == 0 and current_gradient != "-":
            max_gradient = -0.00015
            max_epochs = 1500
            if current_gradient > max_gradient:
                print()
                break
            if epoch >= max_epochs:
                print()
                break

    epoch_list = [i + 1 for i in range(epoch)]

    if save_file is not None:
        torch.save(model.state_dict(), save_file)

    plt.plot(epoch_list, loss_list, color="red")
    plt.title(
        f"Guided Training Loss - {str(loss_list[-1])[:7]} (p:{len(vocab.predicates)}, c:{len(vocab.constants)}, a:{vocab.maxArity}, e:{embed_size})"
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Loss", fontsize=14)
    plt.grid(True)
    plt.savefig(
        f"guided_loss-{embed_type}{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.png"
    )
    print("Saved training loss figure.")

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "-t",
        "--training_file",
        default="mr_train_examples.csv",
        help="File path for the training data (goal/rule/score)"
    )
    aparser.add_argument(
        "-s",
        "--save_model",
        default="mr_model.pt",
        help="File path to save the trained model. Defaults to mr_model.pt."
    )
    aparser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train. Default: 1000"
    )
    aparser.add_argument(
        "--embed_type",
        choices=["unification", "autoencoder", "chainbased", "termwalk"],
        default="unification",
        help="Type of embedding",
    )
    aparser.add_argument(
        "--vocab_file", default="vocab", help="Path to dave generated vocab to."
    )
    aparser.add_argument("-e", "--embed_size", type=int, default=50,
                         help="Embed size. Defaults to 50")
    aparser.add_argument("--embed_model_path", default="rKB_model.pth",
                         help="Path to read a trained embedding model from")
    aparser.add_argument(
        "--train_scoring_model",
        action="store_true",
        help="Flag to enable training the scoring model"
    )
    aparser.add_argument(
        "--kb_path",
        type=str,
        help="Path to the knowledge base file"
    )

    args = aparser.parse_args()

    vocab = Vocabulary()
    vocab.init_from_vocab(args.vocab_file)
    embed_size = args.embed_size

    print("States from vocab: " + args.vocab_file)
    vocab.print_summary()
    print()

    default_save_files = {
        "unification": "uni_mr_model.pt",
        "autoencoder": "auto_mr_model.pt",
        "chainbased": "cb_mr_model.pt",
        "termwalk": "tw_mr_model.pt"
    }

    if args.save_model == "mr_model.pt":
        args.save_model = default_save_files[args.embed_type]

    train_reasoning_model(
        args.training_file, args.num_epochs, args.save_model, vocab, args.embed_type, args.embed_model_path, embed_size=args.embed_size
    )
