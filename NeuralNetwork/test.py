import numpy as np

y_pred = np.array([
    [0.1, 0.7, 0.2],
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3]
])
y_true = np.array([1, 0, 2])  # correct class indices

# select the correct predictions
correct_confidences = y_pred[range(len(y_pred)), y_true]

print(correct_confidences)
# calculate loss per sample
losses = -np.log(correct_confidences)

# mean loss
data_loss = np.mean(losses)
print(losses)
