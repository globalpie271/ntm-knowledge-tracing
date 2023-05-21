from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from ntm.ntm import NTM
import pandas as pd
import numpy as np
import argparse
import random
import glob
import os

import torch.nn.functional as F
import torch.optim as optim
import torch


parser = argparse.ArgumentParser(description="NTM for knowledge tracing")
parser.add_argument("--train", help="Trains the model", action="store_true")
parser.add_argument("--encoder", help="Transformer encoder controller", action="store_true")
parser.add_argument("--fm", help="Factorization machine as last layer with sigmoid", action="store_true")
parser.add_argument("--eval", help="Evaluates the model. Default path is models/kt.pt", action="store_true")
parser.add_argument("--modelpath", help="Specify the model path to load, for training or evaluation", type=str)
parser.add_argument("--epochs", help="Specify the number of epochs for training", type=int, default=100)
args = parser.parse_args()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def get_bundle_id(item_id):
  """
  Returns bundle id using questions dataframe
  Args:
    item_id: item id
  Returns:
    str: bundle id
  """
  item_type = item_id[0]
  
  if item_type == 'b':
    return item_id
  elif item_type == 'q':
    return questions_df[questions_df['question_id']==item_id]['bundle_id'].iloc[0]
  elif item_type == 'e':
    return 'b' + item_id[1:]
  else:
    return None

def get_sparse(categories, length = 2, norm=True):
    """
    Gets sparse representation of categories
    Args:
        categories (list): list with category ids
        length (int): number of categories (vector length)
        norm (bool): whether normalize vector or not (the vector is normalized such that it's components sum up to 1)
    Returns:
        list: sparse list of categories from categories list
    """
    categories_list = np.array(categories) - 1
    categories_feature = np.eye(length)[categories_list] if categories else np.zeros(length)
    categories_feature = categories_feature.sum(axis=0) if categories else np.zeros(length)
    if norm and categories:
        return (categories_feature / (categories_feature.sum())).tolist()
    else:
        return categories_feature.tolist()

def one_hot(values, length):
  """
  One-hot encoding
  """
  return np.eye(length)[values]

def get_features(df, max_elapsed_time):
  """
  Extracts feature arrays from the user's dataframe
  """
  # Merge with questions_df
  df = pd.merge(df, questions_df, left_on = 'item_id', right_on = 'question_id', how = 'left')
  
  # Cum sum using group by bundle id
  df['bundle_id'] = pd.DataFrame({'id': df['item_id'], 'previous_id': df['item_id'].shift()}).apply(
    lambda row: get_bundle_id(row['id']), axis = 1)
  df['action_type_list'] = df['action_type'].apply(lambda x: [x])
  df['actions'] = df.groupby('bundle_id')['action_type_list'].apply(lambda group : group.cumsum())
  
  # Elapsed time encoding
  df['elapsed_time'] = df['timestamp'].diff()
  # Filter by questions
  df = df[df['item_id'].str.startswith('q')]
  elapsed_time = (df['elapsed_time']/max_elapsed_time).to_list()
  
  # Actions encoding
  df = df.dropna(subset = 'actions')
  default_actions_set = set(default_actions.keys())
  df['actions'] = df['actions'].apply(lambda actions: set(actions).intersection(default_actions_set))
  df['actions'] = df['actions'].apply(lambda actions: [default_actions[action] for action in actions])
  actions = df['actions'].apply(lambda actions: get_sparse(actions, len(default_actions))).to_list()
  del df['action_type_list']
  
  # Source encoding
  source_categories, _ = pd.factorize(df['source'], sort = True)
  source = df['source'].apply(lambda source: get_sparse([default_sources[source]], len(default_sources))).to_list()  
  
  # Tags encoding
  df['tags'] = df['tags'].str.split(';').apply(lambda tags_list: [tags_map[int(tag)] for tag in tags_list])
  tags = df['tags'].apply(lambda tags: get_sparse(tags, len(tags_map))).to_list()
  # Correction encoding
  correctness = np.float32(df['user_answer'] == df['correct_answer'])
  return np.array(actions), np.array(source), np.array(tags), np.array(elapsed_time)[..., np.newaxis], correctness


# Preprocess data
questions_df = pd.read_csv(os.path.join('data', 'questions.csv'))
csv_paths = glob.glob(os.path.join('data', 'KT4', '*.csv'))

i, limit=0, 100
dfs = dict()
for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    if not df['user_answer'].isna().all():
      user_id = os.path.splitext(os.path.basename(csv_path))[0]
      dfs[csv_path] = df
      dfs[csv_path]['user_id'] = user_id
      i+=1
    if i==limit:
      break

dataset_df = pd.concat(dfs.values())
dataset_df = pd.merge(dataset_df, questions_df, left_on = 'item_id', right_on = 'question_id', how = 'left')
default_actions = {action_type: i for i, action_type in enumerate(dataset_df.action_type.dropna().unique())}
default_sources = {source_type: i for i, source_type in enumerate(dataset_df.source.dropna().unique())}

tags = set()
dataset_df[dataset_df['item_id'].str.startswith('q')]['tags'].str.split(';').apply(lambda tags_list: [tags.add(int(tag)) for tag in tags_list])
tags_map = dict(zip(sorted(tags), range(1, len(tags) + 1)))

def get_ednet(users = 100):
  """
  Get EdNet features
  """
  X, Y = [], []
  i = 0
  for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    if not df['user_answer'].isna().all():
      actions, source, tags, elapsed_time, correctness = get_features(df, 40_000)
      # x = np.hstack([actions, source, tags, elapsed_time])
      # x = np.hstack([actions, source, elapsed_time])
      x = np.hstack([actions, elapsed_time])
      X.append(x)
      Y.append(correctness[..., np.newaxis])
      i+=1
    if i==users:
      break
  return X, Y

def test_eval(X,Y, model, device = None):
  if device is None:
     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  results = []
  y_outs = []
  for x, y in zip(X, Y):
    x = torch.from_numpy(x).float()
    target = torch.from_numpy(y).float()
    x, target = x.to(device), target.to(device)
    batch_size = x.shape[0]
    state = model.get_initial_state(batch_size)
    y_out, state = model(x, state)
    y_out_binarized = y_out.clone().data
    y_outs.append(y_out.clone().data)
    y_out_binarized = torch.where(y_out_binarized<0.5, 0, 1)
    results.extend((y_out_binarized == target).squeeze(1).tolist())
  # Metrics
  threshold = 0.5
  y_outs, y_flatten = torch.cat(y_outs, 0).flatten().cpu().numpy(), torch.Tensor(np.concatenate(Y, 0)).flatten()
  acc = accuracy_score(y_flatten, y_outs>threshold)
  auc = roc_auc_score(y_flatten, y_outs)
  precision = precision_score(y_flatten, y_outs>threshold)
  recall = recall_score(y_flatten, y_outs>0.5)
  return acc, precision, recall, auc
  
def eval(model_path):
  # Define device
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  # Define data
  train_length, length = 10, 100
  X, Y = get_ednet(length)

  # Define params
  vector_length = X[0].shape[-1]
  memory_size = (10, 10)
  hidden_layer_size = 10
  lstm_controller = not args.encoder
  fm_activation = args.fm
  
  # Define model
  model = NTM(vector_length, hidden_layer_size, memory_size, 1, lstm_controller, output_layer='fm' if fm_activation else 'fc')
  optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
  # optimizer = optim.Adam(model.parameters())
  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint)
  model.to(device)
  acc, precision, recall, auc = test_eval(X[train_length:], Y[train_length:], model, device)
  print(f"Test metrics: {[acc, precision, recall, auc]}")
  

def train(epochs=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_length, length = 10, 100
    X, Y = get_ednet(length)
    X_train = np.concatenate(X[:train_length], axis = 0)
    target = np.concatenate(Y[:train_length], axis = 0)

    X_train = torch.from_numpy(X_train).float()
    target = torch.from_numpy(target).float()
    X_train, target = X_train.to(device), target.to(device)
    print(X_train.shape)

    sequence_min_length = 1
    sequence_max_length = 20
    vector_length = X[0].shape[-1]
    memory_size = (10, 10)
    hidden_layer_size = 10
    lstm_controller = not args.encoder
    fm_activation = args.fm
    
    # Define model
    model = NTM(vector_length, hidden_layer_size, memory_size, 1, lstm_controller, output_layer='fm' if fm_activation else 'fc')
    optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
    # optimizer = optim.Adam(model.parameters())
    feedback_frequency = 10

    os.makedirs("models", exist_ok=True)
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    model.to(device)
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        try:
          state = model.get_initial_state(X_train.shape[0])
          y_out, state = model(X_train, state)
          loss = F.binary_cross_entropy(y_out, target)
          loss.backward()
          optimizer.step()
          if epoch % feedback_frequency == 0:
            print(f"Loss at step {epoch}: {loss.item()}")
            acc, precision, recall, auc = test_eval(X[train_length:], Y[train_length:], model, device)
            print(f"Test metrics at {epoch}: {[acc, precision, recall, auc]}")
            model_name, model_type = os.path.splitext(model_path)
            model_epoch_path = model_name + f'_{epoch}'+ model_type
            torch.save(model.state_dict(), model_epoch_path)    
            print(f'{model_epoch_path} saved')
        except KeyboardInterrupt:
           break
    torch.save(model.state_dict(), model_path)
    print(f'{model_epoch_path} saved')

if __name__ == "__main__":
    model_path = "models/kt.pt"
    if args.modelpath:
        model_path = args.modelpath
    if args.train:
        train(args.epochs)
    if args.eval:
        eval(model_path)
