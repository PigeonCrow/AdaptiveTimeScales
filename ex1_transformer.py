# Transformer, trained on  subjects responses in the predictive inference task

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DATA = "data/data_trial_preprocessed.csv"


N_TRIALS = 240
FIT_END = 120          

HIDDENL = 8 
#HIDDENL = 16 
#HIDDENL = 32 
#HIDDENL = 64 


OUT = f"transformer_out_{HIDDENL}"

LEARNING_RATE = 0.01

EPOCHS = 500
SEED = 0              




# outcome and response: one row per session, one column per trial.
# The response on a trial is subjects prediction, given before the outcome of that trial is shown.
def load_data():

    table = pd.read_csv(DATA, usecols=["ID", "session", "trials", "outcome", "response"])
    
    trials = table.sort_values(["ID", "session", "trials"])

    sessions = trials[["ID", "session"]].drop_duplicates().reset_index(drop=True)

    outcome = trials["outcome"].to_numpy().reshape(len(sessions), N_TRIALS)
    response = trials["response"].to_numpy().reshape(len(sessions), N_TRIALS)

    return sessions, outcome, response



# TODO: review
# step t: the input is the outcome of trial t, the target is the  response on trial t + 1 (both / 100)
def to_tensors(outcome, response):
    inputs = torch.tensor(outcome[:, :-1] / 100, dtype=torch.float32).unsqueeze(-1)
    targets = torch.tensor(np.nan_to_num(response[:, 1:]) / 100, dtype=torch.float32)
    answered = torch.tensor(~np.isnan(response[:, 1:]))
    return inputs, targets, answered


# network 
# one attention layer 
# mask hides later trials, and every trial further back
# loses `slope` from its attention score (memory of about 1 / slope trials).
class TransformerNetwork(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.embed = nn.Linear(1, HIDDENL)
        self.transformer = nn.TransformerEncoderLayer(d_model=HIDDEN, nhead=1, dim_feedforward=HIDDEN, dropout=0.0, batch_first=True)
        self.output = nn.Linear(HIDDEN, 1)
        self.log_slope = nn.Parameter(torch.zeros(()))   # slope = exp(log_slope), starts at 1


    def forward(self, inputs):

        steps = torch.arange(inputs.shape[1])
        
        steps_back = (steps[:, None] - steps[None, :]).float()
        recency = -self.log_slope.exp() * steps_back
        mask = recency.masked_fill(steps_back < 0, float("-inf"))
        hidden = self.transformer(self.embed(inputs), src_mask=mask)
        
        
        return self.output(hidden).squeeze(-1)


# training
def train(network, inputs, targets, answered):

    train_inputs = inputs[:, :FIT_END - 1]
    train_targets = targets[:, :FIT_END - 1]
    train_answered = answered[:, :FIT_END - 1]

    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):

        prediction = network(train_inputs)
        error = (prediction - train_targets)[train_answered]
        loss = (error ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch % 50 == 0:
        #     print(f"epoch {epoch}: error on the training trials {100 * loss.item() ** 0.5:.2f} points")


# predictions for trials 2 to 240, back on the 0 to 100 scale
def predict(network, inputs):
   
    with torch.no_grad():
        
        return 100 * network(inputs).numpy()


# error ->  delta prediction response in points root mse
def print_errors(prediction, outcome, response):

    error = prediction - response[:, 1:]
    baseline = outcome[:, :-1] - response[:, 1:]   # for comparison: "response = the last outcome"

    fit_error = np.sqrt(np.nanmean(error[:, :FIT_END - 1] ** 2))
    test_error = np.sqrt(np.nanmean(error[:, FIT_END - 1:] ** 2))
    baseline_error = np.sqrt(np.nanmean(baseline[:, FIT_END - 1:] ** 2))

    print(fit_error)
    print(test_error)
    print(baseline_error)


# one row per trial (trials 2 to 240)
def save_predictions(sessions, prediction, path):

    table = pd.DataFrame({
        
        "ID": np.repeat(sessions["ID"].to_numpy(), N_TRIALS - 1),
        "session": np.repeat(sessions["session"].to_numpy(), N_TRIALS - 1),
        "trials": np.tile(np.arange(2, N_TRIALS + 1), len(sessions)),
        "prediction": prediction.ravel()
       
        })
    
    table.to_csv(path, index=False)


# one session: outcomes, subjecs responses and the models predictions
def plot_session(sessions, outcome, response, prediction, row, path):

    trials = np.arange(1, N_TRIALS + 1)

    plt.figure(figsize=(11, 4))
    plt.axvspan(FIT_END + 0.5, N_TRIALS + 0.5, color="whitesmoke", label="test trials")
    plt.scatter(trials, outcome[row], s=10, color="gray", label="outcome")
    plt.plot(trials, response[row], color="royalblue", label="human")
    plt.plot(trials[1:], prediction[row], color="red", label="transformer")

    plt.xlabel("trial")

    plt.ylabel("value (0 to 100)")

    plt.title(f"subject {sessions["ID"][row]:g}, session {sessions["session"][row]:g}")
    plt.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    os.makedirs(OUT, exist_ok=True)

    # load data
    sessions, outcome, response = load_data()
    inputs, targets, answered = to_tensors(outcome, response)
    print(f"{len(sessions)} sessions of {N_TRIALS} trials from {sessions["ID"].nunique()} people")

    # network
    torch.manual_seed(SEED)
    network = TransformerNetwork()

    # training
    train(network, inputs, targets, answered)
    slope = network.log_slope.exp().item()
    print(f"learned slope {slope:.2f}: memory of about {1 / slope:.1f} trials")

    # prediction and score
    prediction = predict(network, inputs)
    print_errors(prediction, outcome, response)

    # output and plot 
    torch.save(network.state_dict(), f"{OUT}/transformer_weights.pt")
    save_predictions(sessions, prediction, f"{OUT}/transformer_predictions.csv")
    plot_session(sessions, outcome, response, prediction, row=0, path=f"{OUT}/transformer_session.png")
