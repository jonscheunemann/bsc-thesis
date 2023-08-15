import numpy as np


def split_into_batches(data, batch_size):
    num_samples = len(data["inputs"])
    num_batches = int(np.ceil(num_samples / batch_size))

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch = {
            "inputs": data["inputs"][start_idx:end_idx],
            "targets": data["targets"][start_idx:end_idx]
        }
        batches.append(batch)

    return batches


def reshape_future_prediction(data, history_length, prediction_length):
    inputs, targets = [], []

    for i in range(len(data["inputs"]) - history_length - prediction_length):
        inputs.append(np.concatenate(data["inputs"][i:i + history_length]))
        targets.append(data["targets"][i + history_length + prediction_length])

    return {"inputs": np.array(inputs), "targets": np.array(targets)}
