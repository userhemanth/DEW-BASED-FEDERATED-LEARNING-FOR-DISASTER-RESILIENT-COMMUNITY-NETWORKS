# src/dew_aggregator.py
import os
import numpy as np
import torch
import flwr as fl
from train_model import DisasterCNN

os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/global_model.pth"
NUM_CLASSES = 9  # adjust if you change classes

def fitres_to_ndarrays(fit_res):
    """Convert a FitRes.parameters (Parameters) to a list of numpy ndarrays."""
    return fl.common.parameters_to_ndarrays(fit_res.parameters)

def weighted_fedavg(ndarrays_list, num_examples):
    """
    ndarrays_list: list of lists of ndarrays (one inner list per client)
    num_examples: list or array of client example counts
    returns: list of aggregated ndarrays
    """
    num_clients = len(ndarrays_list)
    if num_clients == 0:
        return None

    # ensure arrays are numpy arrays and float64 for stability
    num_examples = np.array(num_examples, dtype=np.float64)
    total = np.sum(num_examples)
    if total == 0:
        # fallback to simple mean
        total = float(len(num_examples))
        num_examples = np.ones_like(num_examples, dtype=np.float64)

    # number of parameter arrays
    n_params = len(ndarrays_list[0])
    aggregated = []
    for p_idx in range(n_params):
        # stack param p_idx from all clients and do weighted average
        stacked = np.stack([ndarrays_list[c][p_idx].astype(np.float64) for c in range(num_clients)], axis=0)
        weights = (num_examples / total).reshape((-1, 1))
        # If param is not 1D, broadcast weights over param shape
        weights = weights.reshape((weights.shape[0],) + (1,) * (stacked.ndim - 1))
        avg = np.sum(stacked * weights, axis=0)
        aggregated.append(avg.astype(np.float32))
    return aggregated

class DewStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        """
        results: list of tuples (client_proxy, FitRes)
        FitRes has attributes: .parameters, .num_examples, .metrics (maybe)
        Return: (Parameters, {}) or (None, {})
        """
        if not results:
            print("[DEW] No results to aggregate.")
            return None, {}

        # Extract FitRes objects (second element of tuple). Some Flower versions may return different shapes,
        # so we handle a few possibilities robustly.
        fit_res_list = []
        num_examples = []
        metrics_list = []
        for item in results:
            # item can be (client_proxy, fit_res) or directly fit_res depending on Flower internals
            try:
                if len(item) == 2:
                    _, fit_res = item
                elif len(item) == 3:
                    # sometimes server provides (cid, fit_res, something)
                    _, fit_res, _ = item
                else:
                    fit_res = item
            except Exception:
                fit_res = item

            # ensure fit_res has expected attributes
            if not hasattr(fit_res, "parameters"):
                print("[DEW] Warning: encountered result without 'parameters' attribute, skipping.")
                continue

            fit_res_list.append(fit_res)
            # some FitRes objects have num_examples attribute, else try 'num_examples' key in .metrics or default 1
            n = getattr(fit_res, "num_examples", None)
            if n is None:
                # fallback - many FitRes have num_examples field; if not, default to 1
                n = 1
            num_examples.append(n)
            metrics_list.append(getattr(fit_res, "metrics", {}))

        if len(fit_res_list) == 0:
            print("[DEW] No valid FitRes entries to aggregate.")
            return None, {}

        # Convert parameters -> ndarrays per client
        ndarrays_per_client = [fitres_to_ndarrays(fr) for fr in fit_res_list]

        # Weighted federated average
        aggregated_ndarrays = weighted_fedavg(ndarrays_per_client, num_examples)

        # Convert aggregated ndarrays back to Parameters
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_ndarrays)

        # Save aggregated model as PyTorch state_dict for later use / evaluation
        try:
            ndarrs = aggregated_ndarrays
            model = DisasterCNN(num_classes=NUM_CLASSES)
            state_dict = model.state_dict()
            # map the ndarrays to state_dict keys in order
            for k, arr in zip(state_dict.keys(), ndarrs):
                # ensure shapes match; if not, try to reshape safely (best-effort)
                try:
                    state_dict[k] = torch.tensor(arr)
                except Exception:
                    state_dict[k] = torch.tensor(arr.copy().reshape(state_dict[k].shape))
            model.load_state_dict(state_dict, strict=False)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[DEW] Saved aggregated global model -> {MODEL_PATH}")
        except Exception as e:
            print(f"[DEW] Could not save aggregated model: {e}")

        # Optionally compute and print simple average metrics
        try:
            avg_metrics = {}
            # Example: average accuracy across clients if available
            accs = [m.get("accuracy") for m in metrics_list if isinstance(m, dict) and "accuracy" in m]
            if len(accs) > 0:
                avg_metrics["accuracy"] = float(np.mean(accs))
            print(f"[DEW] Round {rnd} aggregated metrics: {avg_metrics}")
        except Exception:
            pass

        # Return Parameters object (what Flower server expects) and empty dict
        return aggregated_parameters, {}

# Start the dew aggregator
if __name__ == "__main__":
    print("🌤️  Starting Dew Aggregator on port 9090...")
    strategy = DewStrategy(min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3)
    fl.server.start_server(server_address="localhost:9090", strategy=strategy)
