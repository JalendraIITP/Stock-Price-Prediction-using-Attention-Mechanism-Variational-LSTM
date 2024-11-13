import os
import torch
import joblib
import argparse
import warnings
import numpy as np
import yfinance as yf
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from LSTM_Models import AMVLSTM, LSTM, VLSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name):
  """
    Create a Instance of Model and Load the Pretrained Weights
  """
  model = None
  if model_name == 'LSTM':
    model = LSTM()
  elif model_name == 'VLSTM':
    model = VLSTM()
  else:
    model = AMVLSTM()

  model_file = os.path.join("Models", model_name, f"{model_name}_model.pth")
  model.load_state_dict(torch.load(model_file))
  return model

def load_data(StockName, startDate, endDate, sc):
  """
    Load the Stock Data, Normalize the data, Create Sequence for Prediction
  """
  startDate = datetime.strptime(startDate, "%Y-%m-%d").date()
  endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

  test_data = yf.download(StockName, start=startDate, end=endDate)
  x_test = test_data['Adj Close'].values.reshape(-1, 1)

  test_scaled = sc.transform(x_test)
  
  def create_sequences(data, seq_length=60):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0].reshape(1,seq_length))
        y.append(data[i, 0].reshape(-1))
    return np.array(x), np.array(y)

  x_test, y_test = create_sequences(test_scaled)
  x_test = torch.tensor(x_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  test_loader = torch.utils.data.DataLoader(list(zip(x_test, y_test)), batch_size=32, shuffle=False)
  return test_loader

def test(model, test_loader):
  """
    Test the Model on the Test Data and Calculate Metrics
  """
  model.eval()
  predictions = []
  true_labels = []

  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)

      predictions.extend(outputs.cpu().detach().numpy())
      true_labels.extend(labels.cpu().detach().numpy())

  return np.array(predictions), np.array(true_labels)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--StockName', type=str, required=True)
  parser.add_argument('--startDate', type=str, required=True)
  parser.add_argument('--endDate', type=str, required=True)
  parser.add_argument('--Model_Name', type=str, required=True)

  args = parser.parse_args()
  
  sc_path = os.path.join("Models", args.Model_Name)
  scaler_path = os.path.join(sc_path, "MinMaxScaler.pkl")
  sc = joblib.load(scaler_path)

  model = load_model(args.Model_Name).to(device)

  test_loader = load_data(args.StockName, args.startDate, args.endDate, sc)
  y_pred, y_true = test(model, test_loader)

  y_pred = sc.inverse_transform(y_pred)
  y_true = sc.inverse_transform(y_true)

  """
  Calculate and Print Metrics for the Model
  """
  print('\n')
  print(f"Evaluation Table for {args.Model_Name}")
  r2_LSTM = r2_score(y_true, y_pred)
  MSE_LSTM = mean_squared_error(y_true, y_pred)
  MAE_LSTM = mean_absolute_error(y_true, y_pred)
  print("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
  print("| {:<18} | {:<13} | {:<13} | {:<13} |".format("Model", "R2 Score", "MSE", "MAE"))
  print("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
  print("| {:<18} | {:<13.4f} | {:<13.4f} | {:<13.4f} |".format(args.Model_Name, r2_LSTM, MSE_LSTM, MAE_LSTM))
  print("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
  print("\n")

  results_dir = "Results"
  os.makedirs(results_dir, exist_ok=True)

  """
  Plot the results and Save the results
  """
  plt.figure(figsize=(14, 7))
  plt.plot(range(len(y_pred)), y_pred, label='Predicted Stock Prices', color='blue', linewidth=1.5)
  plt.plot(range(len(y_true)), y_true, label='True Stock Prices', color='red', linewidth=1.5)
  plt.title(f"True Value vs Predicted Value of {args.Model_Name}")
  plt.xlabel('Time')
  plt.ylabel('Stock Price')
  plt.legend()
  plt.grid()

  output_file = os.path.join(results_dir, f"{args.Model_Name}.png")
  plt.savefig(output_file, bbox_inches='tight')
  plt.close()
