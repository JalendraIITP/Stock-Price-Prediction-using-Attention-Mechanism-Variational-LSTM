import os
import torch
import joblib
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import yfinance as yf
import torch.nn as nn
from datetime import datetime
from LSTM_Models import AMVLSTM, LSTM, VLSTM
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(model_name):
  RNN = None
  if model_name == 'AMVLSTM':
    RNN = AMVLSTM
  elif model_name == 'VLSTM':
    RNN = VLSTM
  else:
    RNN = LSTM
  
  """
    Hyperparameters
  """
  input_size = 60
  hidden_size = 128
  num_layers = 1

  np.random.seed(42)
  torch.manual_seed(42)
  model = RNN(input_size, hidden_size, num_layers).to(device)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  return model, criterion, optimizer

def load_data(StockName, model_name, startDate, endDate):
  """
  Load historical data for the given stock and create sequences for training.
  """
  startDate = datetime.strptime(startDate, "%Y-%m-%d").date()
  endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

  train_data = yf.download(StockName, start=startDate, end=endDate)
  X_train = train_data['Adj Close'].values.reshape(-1, 1)
  
  """
  Normalize the data using MinMaxScaler and Save the MinMaxScaler
  """
  sc = MinMaxScaler(feature_range=(0, 1))
  train_scaled = sc.fit_transform(X_train)
  sc_path = os.path.join("Models", model_name)
  os.makedirs(sc_path, exist_ok=True)
  scaler_path = os.path.join(sc_path, "MinMaxScaler.pkl")
  joblib.dump(sc, scaler_path)

  """
  Create sequences for training
  """
  def create_sequences(data, seq_length=60):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0].reshape(1,seq_length))
        y.append(data[i, 0].reshape(-1))
    return np.array(x), np.array(y)

  X_train, y_train = create_sequences(train_scaled)
  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)

  """
  Create DataLoader for training
  """
  train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
  return train_loader

def train(model, train_loader, criterion, optimizer, verbose=1, num_epochs=100):
  model.train()
  model.to(device)
  history = []
  print("Model is Training......................\n")

  # Initialize tqdm progress bar for epochs
  epoch_progress = tqdm(range(num_epochs), desc="Training", unit="epoch")

  for epoch in epoch_progress:
    total_loss = 0.0
    num_batches = 0

    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      num_batches += 1

    average_loss = total_loss / num_batches
    history.append(average_loss)
    
    epoch_progress.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.6f}")
  
  return history

def save_model(model, model_name):
  """
  Save the trained model to a file.
  """
  m_path = os.path.join("Models", model_name)
  os.makedirs(m_path, exist_ok=True)
  model_file = os.path.join(m_path, f"{model_name}_model.pth")
  torch.save(model.state_dict(), model_file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--StockName', type=str, required=True)
  parser.add_argument('--startDate', type=str, required=True)
  parser.add_argument('--endDate', type=str, required=True)
  parser.add_argument('--Model_Name', type=str, required=True)
  args = parser.parse_args()
  
  model, criterion, optimizer = create_model(args.Model_Name)

  train_loader = load_data(args.StockName, args.Model_Name, args.startDate, args.endDate)

  epoch = 100

  history = train(model,train_loader,criterion,optimizer,num_epochs=epoch)
  
  save_model(model, args.Model_Name)
