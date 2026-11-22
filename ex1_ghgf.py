import os
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pyhgf.model import Network
from pytensor import wrap_jax

jax.config.update("jax_enable_x64", True)


DATA = "data/data_trial_preprocessed.csv"
OUT = "ghgf_outputs_nocopy_filter"

N_TRIALS = 240
NOISE_SD = 10   # outcome noise assumed by the model
FIT_END = 120     # fit on trials 2 to 120, test on trials 121 to 240

# outcome and response: one row per session, one column per trial.
# The response on a trial is the person's prediction, given before the outcome of that trial is shown.
def load_data():

    table = pd.read_csv(DATA, usecols=["ID", "session", "trials", "outcome", "response"])

    #is_copy = (table["ID"] == 3) & (table["session"] == 2)   # the same session as subject 3's session 3
    
    #trials = table[~is_copy].sort_values(["ID", "session", "trials"])

    trials = table.sort_values(["ID", "session", "trials"])

    sessions = trials[["ID", "session"]].drop_duplicates().reset_index(drop=True)
    outcome = trials["outcome"].to_numpy().reshape(len(sessions), N_TRIALS)
    response = trials["response"].to_numpy().reshape(len(sessions), N_TRIALS)


    return sessions, outcome, response


# network 
def build_network(first_outcome, omega, theta):

    return (
        Network()
        .add_nodes(precision=1 / NOISE_SD ** 2)                                                     # 0 u
        .add_nodes(value_children=0, mean=first_outcome, precision=1 / 3, tonic_volatility=omega)   # 1 x
        .add_nodes(volatility_children=1, mean=4.0, precision=10.0, tonic_volatility=theta)         # 2 x_v
    )


# predictions for one session, from trial 2 on
def predict_session(omega, theta, outcomes):
    network = build_network(outcomes[0], omega, theta).input_data(input_data=outcomes[1:])
    return network.node_trajectories[1]["expected_mean"]


# fit: omega, theta and eta of every session, from the responses on trials 2 to FIT_END
def fit(sessions, outcome, response):

    fit_response = response[:, 1:FIT_END]
    session_index, trial_index = np.nonzero(~np.isnan(fit_response))   # the responses that were given
    predict_sessions = wrap_jax(jax.vmap(predict_session))              # all sessions at once, callable from PyMC

    with pm.Model():

        omega = pm.Normal("omega", 0.0, 2.0, shape=len(sessions))
        theta = pm.Normal("theta", -6.0, 2.0, shape=len(sessions))
        eta = pm.Normal("eta", 1.0, 2.0, shape=len(sessions))

        prediction = predict_sessions(omega, theta, outcome[:, :FIT_END])

        pm.Normal("response", mu=prediction[session_index, trial_index], sigma=pm.math.exp(eta)[session_index], observed=fit_response[session_index, trial_index])

        estimate = pm.find_MAP(tol=1e-12)   # tol  until the fit no longer improves


    return sessions.assign(omega=estimate["omega"], theta=estimate["theta"], eta=estimate["eta"])


# predictions for every session, trials 2 to 240, with the fitted parameters
def predict(parameters, outcome):
    omega = parameters["omega"].to_numpy()
    theta = parameters["theta"].to_numpy()
    return np.asarray(jax.vmap(predict_session)(omega, theta, outcome))


# error: distance between prediction and response in points (root mean squared error)
def print_errors(prediction, outcome, response):

    error = prediction - response[:, 1:]
    baseline = outcome[:, :-1] - response[:, 1:]   # for comparison: "response = the last outcome"

    fit_error = np.sqrt(np.nanmean(error[:, :FIT_END - 1] ** 2))
    test_error = np.sqrt(np.nanmean(error[:, FIT_END - 1:] ** 2))
    baseline_error = np.sqrt(np.nanmean(baseline[:, FIT_END - 1:] ** 2))

    print(f"error on fit trials  (2 to {FIT_END}): {fit_error:.2f} points")
    print(f"error on test trials ({FIT_END + 1} to {N_TRIALS}): {test_error:.2f} points")
    print(f"error on test trials, last outcome as prediction: {baseline_error:.2f} points")


# one row per trial (trials 2 to 240)
def save_predictions(sessions, prediction, path):

    table = pd.DataFrame({"ID": np.repeat(sessions["ID"].to_numpy(), N_TRIALS - 1),
                          "session": np.repeat(sessions["session"].to_numpy(), N_TRIALS - 1),
                          "trials": np.tile(np.arange(2, N_TRIALS + 1), len(sessions)),
                          "prediction": prediction.ravel()})
    
    table.to_csv(path, index=False)



def plot_session(sessions, outcome, response, prediction, row, path):

    trials = np.arange(1, N_TRIALS + 1)

    plt.figure(figsize=(11, 4))
    plt.axvspan(FIT_END + 0.5, N_TRIALS + 0.5, color="whitesmoke", label="test trials")
    plt.scatter(trials, outcome[row], s=10, color="gray", label="outcome")
    plt.plot(trials, response[row], color="royalblue", label="person")
    plt.plot(trials[1:], prediction[row], color="red", label="gHGF")
    plt.xlabel("trial")
    plt.ylabel("value (0 to 100)")
    plt.title(f"subject {sessions['ID'][row]:g}, session {sessions['session'][row]:g}")
    plt.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_beliefs(parameters, outcome, row, path):

    omega = parameters["omega"][row]
    theta = parameters["theta"][row]
    network = build_network(outcome[row, 0], omega, theta).input_data(input_data=outcome[row, 1:])

    network.plot_trajectories()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    os.makedirs(OUT, exist_ok=True)

    # 1. load data
    sessions, outcome, response = load_data()
    print(f"{len(sessions)} sessions of {N_TRIALS} trials from {sessions['ID'].nunique()} people")

    # 2. build the network + 3. fit
    #    every session has its own parameters, so fit builds one network per session with build_network
    parameters = fit(sessions, outcome, response)
    print(parameters[["omega", "theta", "eta"]].describe().round(2))

    # 4. predict and score
    prediction = predict(parameters, outcome)
    print_errors(prediction, outcome, response)

    # 5. save and plot
    parameters.to_csv(f"{OUT}/ghgf_parameters.csv", index=False)
    save_predictions(sessions, prediction, f"{OUT}/ghgf_predictions.csv")
    plot_session(sessions, outcome, response, prediction, row=0, path=f"{OUT}/ghgf_session.png")
    plot_beliefs(parameters, outcome, row=0, path=f"{OUT}/ghgf_beliefs.png")
