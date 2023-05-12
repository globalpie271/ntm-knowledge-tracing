import random
import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from ntm.ntm import NTM
from ntm.utils import plot_copy_results
import pandas as pd
import numpy as np
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--train", help="Trains the model", action="store_true")
parser.add_argument("--ff", help="Feed forward controller", action="store_true")
parser.add_argument("--eval", help="Evaluates the model. Default path is models/copy.pt", action="store_true")
parser.add_argument("--modelpath", help="Specify the model path to load, for training or evaluation", type=str)
parser.add_argument("--epochs", help="Specify the number of epochs for training", type=int, default=100)
args = parser.parse_args()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

csv_paths = glob.glob(os.path.join('..', 'data', 'KT4', '*.csv'))
with open(os.path.join('..', 'data', 'tags_map.pickle'), 'rb') as handle:
    tags_map = pickle.load(handle)
default_sources = {'review_quiz': 1, 'archive': 2, 'my_note': 3, 'tutor': 4, 'diagnosis': 5, 'adaptive_offer': 6, 'review': 7, 'sprint': 8}
default_actions = {'play_audio': 1, 'play_video': 2}
questions_df = pd.read_csv(os.path.join('..', 'data', 'questions.csv'))
# lectures_df = pd.read_csv(os.path.join('contents', 'lectures.csv'))
def get_bundle_id(item_id, previous_bundle_id):
  """
  Returns bundle id using questions and lectures
  Args:
    item_id: item id
    previous_bundle_id: previous id for lectures that are not connected with bundles
  """
  # if item_id.startswith('l'):
  #   item_id = previous_bundle_id
  
  item_type = item_id[0]
  
  if item_type == 'b':
    return item_id
  elif item_type == 'q':
    return questions_df[questions_df['question_id']==item_id]['bundle_id'].iloc[0]
  elif item_type == 'e':
    return 'b' + item_id[1:]
  else:
    return None

# def get_sparse(skills, skills_num, norm=True):
def get_sparse(categories, length = 2, norm=True):
    """
    Gets sparse representation of categories
    Args:
        categories (list): list with category ids
        skills_num (int): number of categories (vector length)
        norm (bool): whether normalize vector or not (the vector is normalized such that it's components sum up to 1)
    """
    categories_list = np.array(categories) - 1
    categories_feature = np.eye(length)[categories_list] if categories else np.zeros(length)
    categories_feature = categories_feature.sum(axis=0) if categories else np.zeros(length)
    if norm and categories:
        return (categories_feature / (categories_feature.sum())).tolist()
    else:
        return categories_feature.tolist()

def one_hot(values, length):
  return np.eye(length)[values]

def get_features(df, max_elapsed_time):
  # Merge with questions_df
  df = pd.merge(df, questions_df, left_on = 'item_id', right_on = 'question_id', how = 'left')
  # Cum sum using group by bundle id
  df['bundle_id'] = pd.DataFrame({'id': df['item_id'], 'previous_id': df['item_id'].shift()}).apply(
    lambda row: get_bundle_id(row['id'], row['previous_id']), axis = 1)
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
  # df['actions'] = df['actions'].apply(lambda actions: get_sparse(actions, len(default_actions)))
  actions = df['actions'].apply(lambda actions: get_sparse(actions, len(default_actions))).to_list()
  del df['action_type_list']
  # Source encoding
  source_categories, _ = pd.factorize(df['source'], sort = True)
  # df['source'] = one_hot(pd.factorize(df['source'], sort = True)[0], len(default_sources))
  # source = one_hot(pd.factorize(df['source'], sort = True)[0], len(default_sources))
  # source = one_hot(pd.factorize(df['source'], sort = True)[0], len(default_sources))
  source = df['source'].apply(lambda source: get_sparse([default_sources[source]], len(default_sources))).to_list()
  # Tags encoding
  df['tags'] = df['tags'].str.split(';').apply(lambda tags_list: [tags_map[int(tag)] for tag in tags_list])
  # df['tags'] = df['tags'].apply(lambda tags: get_sparse(tags, len(tags_map)))
  tags = df['tags'].apply(lambda tags: get_sparse(tags, len(tags_map))).to_list()
  # Correction encoding
  correctness = np.float32(df['user_answer'] == df['correct_answer'])
  return np.array(actions), np.array(source), np.array(tags), np.array(elapsed_time)[..., np.newaxis], correctness



# def get_training_sequence(sequence_min_length, sequence_max_length, vector_length, batch_size=1):
#     sequence_length = random.randint(sequence_min_length, sequence_max_length)
#     output = torch.bernoulli(torch.Tensor(sequence_length, batch_size, vector_length).uniform_(0, 1))
#     input = torch.zeros(sequence_length + 1, batch_size, vector_length + 1)
#     input[:sequence_length, :, :vector_length] = output
#     input[sequence_length, :, vector_length] = 1.0
#     return input, output


def train(epochs=10):
    tensorboard_log_folder = f"runs/copy-task-{datetime.now().strftime('%Y-%m-%dT%H%M%S')}"
    writer = SummaryWriter(tensorboard_log_folder)
    print(f"Training for {epochs} epochs, logging in {tensorboard_log_folder}")
    sequence_min_length = 1
    sequence_max_length = 20
    vector_length = 199
    memory_size = (128, 20)
    hidden_layer_size = 10
    batch_size = 1
    lstm_controller = not args.ff

    writer.add_scalar("sequence_min_length", sequence_min_length)
    writer.add_scalar("sequence_max_length", sequence_max_length)
    writer.add_scalar("vector_length", vector_length)
    writer.add_scalar("memory_size0", memory_size[0])
    writer.add_scalar("memory_size1", memory_size[1])
    writer.add_scalar("hidden_layer_size", hidden_layer_size)
    writer.add_scalar("lstm_controller", lstm_controller)
    writer.add_scalar("seed", seed)
    writer.add_scalar("batch_size", batch_size)

    model = NTM(vector_length, hidden_layer_size, memory_size, lstm_controller)

    optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
    feedback_frequency = 100
    total_loss = []
    total_cost = []

    X, Y = [], []
    for csv_path in csv_paths[:100]:
        df = pd.read_csv(csv_path)
        # actions, source, tags, elapsed_time, correctness = get_features(df, 1.)
        print(csv_path)
        actions, source, tags, elapsed_time, correctness = get_features(df, 40_000)
        # print(actions.shape, source.shape, tags.shape, elapsed_time.shape)
        x = np.hstack([actions, source, tags, elapsed_time])
        X.append(x)
        Y.append(correctness)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    os.makedirs("models", exist_ok=True)
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    model.to(device)
    for epoch in range(epochs + 1):
        print('Next epoch')
        optimizer.zero_grad()
        for x, y in zip(X[:10 + 1], Y[:10 + 1]):
            # x, y = x[np.newaxis, ...], y[np.newaxis, ..., np.newaxis]
            y = y[..., np.newaxis]
            x = torch.from_numpy(x).float()
            target = torch.from_numpy(y).float()
            # optimizer.zero_grad()
            batch_size = x.shape[0]
            x, target = x.to(device), target.to(device)
            state = model.get_initial_state(batch_size)
            # for vector in x:
            #   _, state = model(vector, state)
            y_out, state = model(x, state)
            # y_out = torch.zeros(target.size())
            # for j in range(len(target)):
                ## y_out[j], state = model(torch.zeros(batch_size, vector_length + 1), state)
                # y_out[j], state = model(torch.zeros(batch_size, vector_length), state)
            loss = F.binary_cross_entropy(y_out, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            y_out_binarized = y_out.clone().data
            # y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
            y_out_binarized = torch.where(y_out_binarized<0.5, 0, 1)
            cost = torch.sum(torch.abs(y_out_binarized - target)) / len(target)
            total_cost.append(cost.item())
        # if epoch % feedback_frequency == 0:
        running_loss = sum(total_loss) / len(total_loss)
        running_cost = sum(total_cost) / len(total_cost)
        print(f"Loss at step {epoch}: {running_loss}")
        if epoch % feedback_frequency == 0:
          writer.add_scalar('training loss', running_loss, epoch)
          writer.add_scalar('training cost', running_cost, epoch)
          total_loss = []
          total_cost = []


    # for epoch in range(epochs + 1):
    #     optimizer.zero_grad()
    #     input, target = get_training_sequence(sequence_min_length, sequence_max_length, vector_length, batch_size)
    #     state = model.get_initial_state(batch_size)
    #     for vector in input:
    #         _, state = model(vector, state)
    #     y_out = torch.zeros(target.size())
    #     for j in range(len(target)):
    #         y_out[j], state = model(torch.zeros(batch_size, vector_length + 1), state)
    #     loss = F.binary_cross_entropy(y_out, target)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss.append(loss.item())
    #     y_out_binarized = y_out.clone().data
    #     y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    #     cost = torch.sum(torch.abs(y_out_binarized - target)) / len(target)
    #     total_cost.append(cost.item())
    #     if epoch % feedback_frequency == 0:
    #         running_loss = sum(total_loss) / len(total_loss)
    #         running_cost = sum(total_cost) / len(total_cost)
    #         print(f"Loss at step {epoch}: {running_loss}")
    #         writer.add_scalar('training loss', running_loss, epoch)
    #         writer.add_scalar('training cost', running_cost, epoch)
    #         total_loss = []
    #         total_cost = []

    torch.save(model.state_dict(), model_path)


def eval(model_path):
    # vector_length = 8
    vector_length = 199
    memory_size = (128, 20)
    # hidden_layer_size = 100
    hidden_layer_size = 10
    lstm_controller = not args.ff

    X, Y = [], []
    for csv_path in csv_paths[:100]:
        df = pd.read_csv(csv_path)
        # actions, source, tags, elapsed_time, correctness = get_features(df, 1.)
        print(csv_path)
        actions, source, tags, elapsed_time, correctness = get_features(df, 40_000)
        # print(actions.shape, source.shape, tags.shape, elapsed_time.shape)
        x = np.hstack([actions, source, tags, elapsed_time])
        X.append(x)
        Y.append(correctness)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = NTM(vector_length, hidden_layer_size, memory_size, lstm_controller)

    print(f"Loading model from {model_path}")
    # checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    results = []
    for x, y in zip(X[:10 + 1], Y[:10 + 1]):
      # x, y = x[np.newaxis, ...], y[np.newaxis, ..., np.newaxis]
      y = y[..., np.newaxis]
      x = torch.from_numpy(x).float()
      target = torch.from_numpy(y).float()
      # optimizer.zero_grad()
      batch_size = x.shape[0]
      # x, target = x.to(device), target.to(device)
      state = model.get_initial_state(batch_size)
      # for vector in x:
      #   _, state = model(vector, state)
      y_out, state = model(x, state)
      y_out_binarized = y_out.clone().data
      y_out_binarized = torch.where(y_out_binarized<0.5, 0, 1)
      print(torch.sum(y_out_binarized == target)/target.shape[0])
      results.extend((y_out_binarized == target).squeeze(1).tolist())
      print(sum(results)/len(results))
      # plot_copy_results(target, y_out, vector_length)
    # lengths = [20, 100]
    # for l in lengths:
    #     sequence_length = l
    #     input, target = get_training_sequence(sequence_length, sequence_length, vector_length)
    #     state = model.get_initial_state()
    #     for vector in input:
    #         _, state = model(vector, state)
    #     y_out = torch.zeros(target.size())
    #     for j in range(len(target)):
    #         y_out[j], state = model(torch.zeros(1, vector_length + 1), state)
    #     y_out_binarized = y_out.clone().data
    #     y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    #     plot_copy_results(target, y_out, vector_length)


if __name__ == "__main__":
    # model_path = "models/copy.pt"
    model_path = "models/kt.pt"
    if args.modelpath:
        model_path = args.modelpath
    if args.train:
        train(args.epochs)
    if args.eval:
        eval(model_path)