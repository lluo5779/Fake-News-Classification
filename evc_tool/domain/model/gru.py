from dataclasses import dataclass, asdict
import numpy as np
import os
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tokenizer.tokenizer import tokenize, TokenSet
import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, logsigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook, tnrange
from typing import Callable, Tuple, Dict


DATA_PATH = './pipeline/data_loader/'
GLOVE_PATH = 'Glove/Data'
SESSION_NOTES_CSV = 'temporal_session_notes_1.csv'
KG_PATH = './pipeline/models_embeddings/kg_features.pkl'

@dataclass(frozen=True)
class Params:
  """Immutable dataclass for holding all modeling parameters.
  
  The functions return parts of the arguments. They are designed to be dumped
  into the associated nn.Module or function with **

  :param save_path: save path for the model (e.g., TokenGRU)
  :param model_type: type of model; one of TOKEN_GRU or VENTURE_GRU
  :param aggregate_type: type of GRU hidden layer aggregation
  :param headless: whether the GRU should pass through final linear layer
                   for returning logsoftmax
  :param add_end_token: whether to add an end of line token to data

  All other parameters map to associated values for the functions that use
  them.

  :Example: nn.GRU(**params.gru())

  """
  #FILEPATH
  save_path: str

  #TYPES
  model_type: int
  aggregate_type: int
  aggregate_type_gru: int
  headless: bool
  add_end_token: bool

  #SAMPLING PARAMS
  oversample: bool
  sample_type: int

  #KG PARAMS
  kg: bool
  n_indices: int

  #EMBEDDING LAYER PARAMS
  num_embeddings: int
  embedding_dim: int
  
  #GRU PARAMS
  input_size: int
  hidden_size: int
  batch_first: bool

  #LINEAR PARAMS
  in_features: int
  out_features: int

  #TRAIN PARAMS
  mask_prob: float
  mask_id: int
  lr_rate: float = 1e-3
  batch_size: int = 64
  epochs: int = 100
  weights: torch.tensor = torch.tensor([0.2, 0.8])
  shuffle: bool = True

  #TOKENIZER PARAMS
  s_ngram: int = 10
  s_step: int = 10


  def embedding(self) -> Dict:
    """Returns a Dict of params for nn.Embedding."""
    return dict(num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim)


  def gru(self) -> Dict:
    """Returns a Dict of params for nn.GRU."""
    return dict(input_size=self.input_size,
                hidden_size=self.hidden_size, 
                batch_first=self.batch_first)


  def gru2(self) -> Dict:
    """Returns a Dict of params for nn.GRU."""
    return dict(input_size=self.hidden_size*3*2, #+self.n_indices,
                hidden_size=self.hidden_size, 
                batch_first=self.batch_first)


  def linear(self):
    """Returns a Dict of params for nn.Linear."""
    return dict(in_features=self.in_features,
                out_features=self.out_features)


  def to_dict(self):
    """Returns the entire param set as a Dict."""
    return asdict(self)


class GenericGRU(nn.Module):
  """Generic class for constructing different GRU types.

  :param params: a Params dataclass object with the params for this model

  :raises: Throws generic Exceptions for aggregate_type's and model_type's that
           don't match the constants expressed below.
  """
  # Aggregate types
  POOL_HIDDEN = 0
  ALL_HIDDEN = 1
  ATTENTION = 2
  END_ONLY = 3
  # Model types
  TOKEN_GRU = 0
  VENTURE_GRU = 1
  STACKED_GRU_LINEAR = 2
  STACKED_GRUS = 3
  def __init__(self, params: Params):
    super(GenericGRU, self).__init__()
    self.params = params
    self.headless = params.headless

    self.mask_gen = torch.distributions.binomial.Binomial(1, probs=params.mask_prob)
    self.reset_list = []
    if params.model_type == GenericGRU.TOKEN_GRU or \
       params.model_type == GenericGRU.STACKED_GRU_LINEAR:
      self.embedding = nn.Embedding(**params.embedding())
      self.reset_list.append(self.embedding)
    elif params.model_type == GenericGRU.VENTURE_GRU:
      pass
    elif params.model_type == GenericGRU.STACKED_GRUS:
      self.embedding = nn.Embedding(**params.embedding())
      self.gru2 = nn.GRU(**params.gru2())
      self.reset_list.extend([self.embedding, self.gru2])
      if params.aggregate_type_gru == GenericGRU.ATTENTION:
        self.attn_linear_Q = nn.Linear(params.hidden_size*5,
                                       params.hidden_size*5)
        self.attn_linear_K = nn.Linear(params.hidden_size*5,
                                       params.hidden_size*5)
        self.attn_linear_V = nn.Linear(params.hidden_size*5,
                                       params.hidden_size*5)
        self.reset_list.extend([self.attn_linear_Q,
                                self.attn_linear_K,
                                self.attn_linear_V])
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(-1)
    else:
      raise Exception(f'Invalid model_type: {params.model_type}')
    if params.aggregate_type == GenericGRU.ATTENTION:
      self.attn_linear_Q = nn.Linear(params.hidden_size*params.s_ngram,
                                     params.hidden_size*params.s_ngram)
      self.attn_linear_K = nn.Linear(params.hidden_size*params.s_ngram,
                                     params.hidden_size*params.s_ngram)
      self.attn_linear_V = nn.Linear(params.hidden_size*params.s_ngram,
                                     params.hidden_size*params.s_ngram)
      self.reset_list.extend([self.attn_linear_Q,
                              self.attn_linear_K,
                              self.attn_linear_V])
      self.dropout = nn.Dropout()
      self.softmax = nn.Softmax(-1)
    if self.params.kg:
      self.kg_linear = nn.Linear(params.n_indices, params.hidden_size*3)
    self.gru = nn.GRU(**params.gru())
    self.reset_list.append(self.gru)
    self.linear = nn.Linear(**params.linear())
    self.reset_list.append(self.linear)
    if params.save_path is not None:
      self.load_state_dict(torch.load(params.save_path))

  def forward(self, x: torch.Tensor):
    """Standard nn.Module forward method.
    
    :param x: input Tensor data
    
    """
    x_ = x.clone()
    if self.training:
      if self.params.kg:
        maskable = x_[:, :, :-self.params.n_indices]
        x_[:, :, :-self.params.n_indices] = self.mask(maskable)
      else:
        x_ = self.mask(x_)
    if self.params.model_type == GenericGRU.STACKED_GRU_LINEAR or \
       self.params.model_type == GenericGRU.STACKED_GRUS:
      x_ = torch.cat([x_[:, 0, :],
                      x_[:, 1, :],
                      x_[:, 2, :],
                      x_[:, 3, :],
                      x_[:, 4, :]])
    n_hidden = self.params.hidden_size
    if self.params.model_type == GenericGRU.TOKEN_GRU or \
       self.params.model_type == GenericGRU.STACKED_GRU_LINEAR or \
       self.params.model_type == GenericGRU.STACKED_GRUS:
      if self.params.kg:
        x_ = self.embedding(x_[:, :-self.params.n_indices])
      else:
        x_ = self.embedding(x_)
    gru_out, final_hidden = self.gru(x_)
    if self.params.aggregate_type == GenericGRU.POOL_HIDDEN:
      avg_pool = adaptive_avg_pool2d(gru_out, (1, n_hidden))
      max_pool = adaptive_max_pool2d(gru_out, (1, n_hidden))
      output = torch.cat([final_hidden[-1],
                          avg_pool.view(-1, n_hidden),
                          max_pool.view(-1, n_hidden)],
                          dim=1)
    elif self.params.aggregate_type == GenericGRU.ALL_HIDDEN:
      output = gru_out.contiguous().view(gru_out.shape[0],
                                         n_hidden*self.params.s_ngram)
    elif self.params.aggregate_type == GenericGRU.ATTENTION:
      hiddens = gru_out.contiguous().view(gru_out.shape[0], -1)
      query = self.attn_linear_Q(hiddens).view(gru_out.shape)
      keys = self.attn_linear_K(hiddens).view(gru_out.shape)
      values = self.attn_linear_V(hiddens).view(gru_out.shape)
      output = self.attention(query, keys, values)
    elif self.params.aggregate_type == GenericGRU.END_ONLY:
      avg_pool = adaptive_avg_pool2d(gru_out, (1, n_hidden))
      output = avg_pool.view(-1, n_hidden)
    else:
      raise Exception(f'Invalid aggregate_type: {self.params.aggregate_type}')
    if self.params.model_type == GenericGRU.STACKED_GRU_LINEAR:
      output = torch.cat(torch.split(output, x.shape[0], dim=0), dim=-1)
    elif self.params.model_type == GenericGRU.STACKED_GRUS:
      embeddings = torch.stack(torch.split(output, x.shape[0], dim=0), dim=1)
      if self.params.kg:
        # kg = x[:, :, -self.params.n_indices:].float()/1000.
        kg = self.kg_linear(x[:, :, -self.params.n_indices:].float()/1000.)
        embeddings = torch.cat([embeddings, kg], dim=-1)
      gru_out2, final_hidden2 = self.gru2(embeddings)
      if self.params.aggregate_type_gru == GenericGRU.POOL_HIDDEN:
        avg_pool = adaptive_avg_pool2d(gru_out2, (1, n_hidden))
        max_pool = adaptive_max_pool2d(gru_out2, (1, n_hidden))
        output = torch.cat([final_hidden2[-1],
                            avg_pool.view(-1, n_hidden),
                            max_pool.view(-1, n_hidden)],
                            dim=1)
      elif self.params.aggregate_type_gru == GenericGRU.ATTENTION:
        hiddens = gru_out2.contiguous().view(gru_out2.shape[0], -1)
        query = self.attn_linear_Q(hiddens).view(gru_out2.shape)
        keys = self.attn_linear_K(hiddens).view(gru_out2.shape)
        values = self.attn_linear_V(hiddens).view(gru_out2.shape)
        output = self.attention(query, keys, values)
    if not self.headless:
      output = self.linear(output)
    return output

  def attention(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor):
    """An attention operation. Should be included within the forward class.
    
    :param query: the query Tensor for attention call; usually the preceding
                  embedding
    :param keys: the keys Tensor for attention call; often preceding embedding
                 but it can be a trainable weight vector
    :param values: the values Tensor for attention call; usually the preceding
                   embedding
                   
    """
    scores = torch.matmul(query,
                          keys.transpose(-2, -1)) / np.sqrt(self.params.hidden_size)
    attn = self.dropout(self.softmax(scores))
    output = torch.matmul(attn, values).view(values.shape[0],
                                             -1)
    return output

  def mask(self, x: torch.Tensor): # produces a side effect of changing x
    """Masks the input tensor in place.

    :param x: input Tensor data

    """
    rand_mask = self.mask_gen.sample(torch.Size(x.shape))
    x[rand_mask == 1] = self.params.mask_id
    return x

  def set_headless(self):
    """Sets the model to headless.
    
    When the model is headless, it skips the last linear layer. Usually when
    generating the embeddings for later processes.
    
    """
    self.headless = True

  def set_withhead(self):
    """Sets the model to use the last linear layer.
    
    When the model includes the head, it uses the last linear layer. Usually
    for training.
    
    """
    self.headless = False

  def reset(self, excludes: Dict = {}):
    """Resets the model and all sub-modules/weights.

    :param excludes: a Dict of layers to exclude.

    """
    for layer in self.reset_list:
      try:
        _ = excludes[layer]
      except KeyError:
        layer.reset_parameters()


class GRUManager(object):
  """Manages training and prediction.
  
  :param model: the pytorch model
  :param params: the Params for the model

  """
  def __init__(self, model: nn.Module, params: Params):
    self.model = model
    self.params = params
    self.logsoftmax = nn.LogSoftmax(dim=-1).cuda()

  def fit(self, x_train: torch.Tensor, y_train: torch.Tensor,
          hyperopt: bool = False) -> None:
    """Fits the model to data in a supervised fashion.

    :param x_train: Tensor of training data
    :param y_train: Tensor of training labels
    :param hyperopt: bool indicating whether the call is within hyperopt loop

    """
    self.model.train()
    criterion = nn.CrossEntropyLoss(self.params.weights.cuda())
    optimizer = torch.optim.Adam(self.model.parameters(), self.params.lr_rate)
    dataloader = to_dataloader(x_train, y_train, self.params.batch_size, self.params.shuffle)
    for epoch in range(self.params.epochs):
      if not hyperopt:
        progress_manager = tqdm_notebook(dataloader, leave=False)
      else:
        progress_manager = dataloader
      for X, y in progress_manager:
        if not hyperopt:
          progress_manager.set_description(f'<<Epoch {epoch}>>')
        X, y = X.cuda(), y.cuda()
        logits = self.model(X)
        loss = criterion(logits, y)
        if not hyperopt:
          progress_manager.set_postfix(loss=f'{loss.item():.2f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  def predict(self, x_test) -> torch.Tensor:
    """Produces model output for testing or embedding generation.

    :param x_test: test Tensor data

    """
    self.model.eval()
    dataloader = DataLoader(x_test,
                            batch_size=x_test.shape[0],
                            shuffle=False)
    X = next(iter(dataloader)).cuda()
    logits = self.model(X)
    if self.model.headless:
      yhat = logits
    else:
      yhat = self.logsoftmax(logits).cpu().data.numpy()
    return yhat


  def fit_predict(self,
                  tokenset: TokenSet,
                  train_ids: torch.Tensor,
                  test_ids: torch.Tensor):
    """Fits to train indices then predicts on test indices.

    Handles oversampling based on params originally input to GRUManager.

    :param tokenset: input data handler
    :param train_ids: filtered Tensor of train venture_ids
    :param test_ids: filtered Tensor of test venture_ids

    """
    self.reset()
    train_tokenset = tokenset.filter_data(ventures=train_ids)
    x_train, y_train = train_tokenset.get_xy()
    test_tokenset = tokenset.filter_data(ventures=test_ids)
    x_test, y_test = test_tokenset.get_xy()
    if self.params.oversample:
      x_train, y_train = train_tokenset.random_oversample(
          self.params.sample_type,
          x_train,
          y_train,
          sessions)
    if self.params.model_type == GenericGRU.STACKED_GRU_LINEAR or \
       self.params.model_type == GenericGRU.STACKED_GRUS:
      n_indices = self.params.n_indices
      if self.params.kg:
        kg, maxes = get_kg()
      else:
        kg = None
      x_train, y_train = assemble_gru_embeddings(x_train,
                                                 train_tokenset,
                                                 kg,
                                                 0.,
                                                 1,
                                                 n_indices)
      x_test, y_test = assemble_gru_embeddings(x_test,
                                               test_tokenset,
                                               kg,
                                               0.,
                                               1,
                                               n_indices)
      if self.params.kg:
        x_train[:, :, -n_indices:] = 1000*x_train[:, :, -n_indices:]/maxes
        x_test[:, :, -n_indices:] = 1000*x_test[:, :, -n_indices:]/maxes
    self.fit(np.int64(x_train), np.int64(y_train))
    return f1_score(np.int64(y_test), np.argmax(self.predict(np.int64(x_test)), axis=1))


  def reset(self):
    """Resets the associated model."""
    self.model.reset()


def get_kg():
  """Loads KG data."""
  with open(KG_PATH, 'rb') as f:
    kg = pickle.load(f)
  kg_indices = []
  for k in kg.keys():
    kg_indices.extend(kg[k])
  return kg, np.expand_dims(np.max(kg_indices, axis=0), axis=0)


def get_kg_xy(tokenset: TokenSet, mask: float):
  """Loads train, test for KG."""
  null = np.array([[] for _ in range(tokenset.get_all_data().shape[0])])
  return assemble_gru_embeddings(null, tokenset, get_kg(), mask, 0, 6)


def assemble_gru_embeddings(embeddings: np.array,
                            tokenset: TokenSet,
                            kg: Dict,
                            mask: int,
                            max_count: int,
                            n_indices: int) -> Tuple[np.array, np.array]:
  """Re-maps GRU embeddings and KG indices to [venture, session, embedding+kg].

  :param embeddings: array of embeddings output from TokenGRU
  :param tokenset: a filtered TokenSet object
  :param kg: the loaded KG Dict
  :param mask: the mask value (e.g., 0, -99)
  :param max_count: the maximum number of windows for a single session;
                    max_count=1 means average embedding windows;
                    max_count=3 means average+max+last_embedding concat
  :param n_indices: the number of KG indices

  :returns: new_aggregate_embeddings, new_labels

  """
  embeddings_ = embeddings
  s_embedding = max_count*embeddings.shape[-1]
  if kg is not None:
    s_embedding += n_indices
  empty = np.ones([5, s_embedding])*mask
  ventures = tokenset.get_venture_set()
  venture_pos = tokenset.get_all_ventures().numpy()
  session_pos = tokenset.get_all_sessions().numpy()
  labels_pos = tokenset.get_all_labels().numpy()
  output = []
  label_output = []
  for venture in ventures:
    venture_data = empty.copy()
    venture_bool = venture_pos == venture
    label_output.append(labels_pos[venture_bool][0])
    for session in range(1, 6):
      session_bool = session_pos == session
      overlap = np.logical_and(venture_bool, session_bool)
      if np.equal(np.sum(overlap), 0.):
        continue
      else:
        embedding = embeddings_[overlap].flatten()
        venture_data[session-1, :embedding.shape[0]] = embedding
      if kg is not None:
        try:
          venture_kg = kg[str(int(venture))]
        except KeyError:
          pass
        else:
          venture_data[session-1, -n_indices:] = \
              np.array(venture_kg[session-1])
    output.append(venture_data)
  return np.float32(np.stack(output, axis=0)), \
         np.array(label_output, dtype=np.int64)


def optimizer(parameter_space: Dict) -> float:
  """Optimizer function for hyperopt training of TOKEN_GRU.

  :param parameter_space: Dict of hyperopt parameters

  :returns: negative f1_score

  """
  params = make_params(model_type=GenericGRU.TOKEN_GRU, **parameter_space)
  tokenset = tokenize(os.path.join(DATA_PATH, SESSION_NOTES_CSV), **params.to_dict())
  rs = ShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=17)
  ventures = tokenset.get_venture_set()
  train_inds, test_inds = next(iter(rs.split(ventures)))
  fit_predict = make_TokenGRU(params)
  return -fit_predict(tokenset, ventures[train_inds], ventures[test_inds])


def evaluate(tokenset: TokenSet,
             fit_predict: Callable[[np.array,
                                    np.array], np.float64]) -> np.array:
  """KFold evaluates a model with a fit_predict function.

  Assumes that fit_predict signature accepts tokenset and the venture_ids for
  the training and test sets.

  :param tokenset: a TokenSet of full data
  :param fit_predict: a function that trains a model and returns an f1_score

  """
  kfolds = KFold(3, True)
  f1_scores = []
  ventures = tokenset.get_venture_set()
  for i, (train_inds, test_inds) in enumerate(kfolds.split(ventures)):
    print(f'Processing fold {i}')
    f1 = fit_predict(tokenset, ventures[train_inds], ventures[test_inds])
    print(f'Fold {i} F1 score: {f1}')
    f1_scores.append(f1)
  return np.array(f1_scores)


def evaluate_kg(tokenset: TokenSet,
                tokenset_full: TokenSet,
                kg: Dict,
                token_manager: GRUManager,
                venture_fit: Callable[[torch.Tensor, torch.Tensor], None],
                venture_predict: Callable[[torch.Tensor], np.array],
                venture_manager: GRUManager,
                max_count: int = None,
                mask: int = 0,
                n_indices: int = 6) -> float:
  """Evaluates a two-step stacked Token-Venture GRU.

  Handles the training of the TokenGRU on the training set, producing the
  embeddings, re-mapping the embeddings and training the VentureGRU.

  :param tokenset: an unfiltered TokenSet of data
  :param tokenset_full: an unfiltered TokenSet of data with each session
                        encoded as a single token window
  :param kg: the loaded KG Dict
  :param token_manager: GRUManager for TokenGRU
  :param venture_manager: GRUManager for VentureGRU
  :param max_count: the maximum number of windows for a single session;
                    max_count=1 means average embedding windows;
                    max_count=3 means average+max+last_embedding concat
  :param mask: the mask value (e.g., 0, -99)
  :param n_indices: the number of KG indices

  """
  kfolds = KFold(3, True)
  f1_scores = []
  ventures = tokenset.get_venture_set()
  if max_count is None:
    counts = tokenset.get_counts(range(1, 6))
    max_count = counts[TokenSet.MAX]
  for i, (train_inds, test_inds) in enumerate(kfolds.split(ventures)):
    token_manager.reset()
    token_manager.model.set_withhead()
    venture_manager.reset()
    print(f'Processing fold {i}')
    v_train_inds = ventures[train_inds]
    v_test_inds = ventures[test_inds]
    train_tokenset = tokenset.filter_data(ventures=v_train_inds)
    train_tokenset_full = tokenset_full.filter_data(ventures=v_train_inds)
    test_tokenset = tokenset_full.filter_data(ventures=v_test_inds)
    x_train, y_train = [np.int64(d) for d in train_tokenset.get_xy()]
    x_train_full, y_train_full = [np.int64(d)
                                  for d in train_tokenset_full.get_xy()]
    x_test, y_test = [np.int64(d) for d in test_tokenset.get_xy()]
    print('Processing TokenGRU')
    token_manager.fit(x_train, y_train)
    token_manager.model.set_headless()
    raw_train_embeddings = token_manager.predict(
        x_train_full).data.cpu().numpy()
    raw_test_embeddings = token_manager.predict(x_test).data.cpu().numpy()
    if kg is not None:
      kg_train = raw_train_embeddings[:, -6:]
      scaler = MinMaxScaler(feature_range=(-1, 1))
      scaler.fit(kg_train)
      raw_train_embeddings[:, -6:] = scaler.transform(kg_train)
      kg_test = raw_test_embeddings[:, -6:]
      scaler = MinMaxScaler(feature_range=(-1, 1))
      scaler.fit(kg_test)
      raw_test_embeddings[:, -6:] = scaler.transform(kg_test)
    print('Aggregating embeddings')
    venture_x_train, venture_y_train = assemble_gru_embeddings(
        raw_train_embeddings,
        train_tokenset_full,
        kg,
        mask,
        max_count,
        n_indices)
    venture_x_test, venture_y_test = assemble_gru_embeddings(
        raw_test_embeddings,
        test_tokenset,
        kg,
        mask,
        max_count,
        n_indices)
    print('Processing VentureGRU')
    venture_manager.fit(venture_x_train, venture_y_train)
    f1 = f1_score(venture_y_test,
                  np.argmax(venture_manager.predict(venture_x_test), axis=1))
    f1_scores.append(f1)
  return np.array(f1_scores)


def get_tokens(params, **kwargs):
  """Convenience function for getting data, labels from params.

  Handles oversampling based on Params object.

  :param params: a Params object with model params

  """
  tokenset = tokenize(os.path.join(DATA_PATH, SESSION_NOTES_CSV), **params.to_dict())
  if params.oversample:
    data, labels = tokenset.random_oversample(params.sample_type, **kwargs)
  else:
    data = tokenset.get_all_data()
    labels = tokenset.get_all_labels()
  return data, labels


def make_TokenGRU(params):
  """Convenience function to make fit_predict for TokenGRU.

  :param params: a Params object for TokenGRU

  """
  model = GenericGRU(params).cuda()
  manager = GRUManager(model, params)
  return manager.fit_predict


def make_GRUs(token_params, venture_params):
  """Makes the GRUManagers for TokenGRU and VentureGRU.

  :param token_params: a Params object for TokenGRU
  :param venture_params: a Params object for VentureGRU

  """
  token_model = GenericGRU(token_params).cuda()
  venture_model = GenericGRU(venture_params).cuda()
  token_manager = GRUManager(token_model, token_params)
  venture_manager = GRUManager(venture_model, venture_params)
  return token_manager, venture_manager


def make_params(model_type,
                tokenset,
                aggregate_type, # choice [0, 1]
                add_end_token,  # choice [False, True]
                oversample,     # choice [False, True]
                sample_type,    # choice [0, 1]
                embedding_dim,  # choice [4, 8, 16, 32, 64]
                hidden_size,    # choice [4, 8, 16, 32, 64]
                mask_prob,      # uniform [0, 1]
                lr_rate,        # loguniform [-10, -1]
                batch_size,     # choice [32, 64, 128]
                epochs,         # qloguniform [3, 6, 10]
                minority_weight,# uniform [0, 1]
                s_ngram,        # quniform [4, 10, 2]
                step_scale,     # uniform [0, 1]
                headless=False,
                kg=False,
                aggregate_type_gru=0, # choice [0, 1]
                save_path=None,
                **kwargs):
  """Convenience function for handling logic for making Params object.

  Most of the args are outlined in the Params class.

  :param embedding_dim: the size of the embedding and the input_size to GRU
  :param minority_weight: weighting for the under-represented class; the
                          the majority class is 1 - minority_weight
  :param step_scale: value between [0, 1] that maps to s_step by scaling
                     s_ngram by this value then taking the closest integer

  """
  num_embeddings = int(tokenset.word2idx['_END'] + 1)
  mask_id = int(tokenset.word2idx['_MASK'])
  if aggregate_type == GenericGRU.ALL_HIDDEN:
    in_features = hidden_size*s_ngram
  elif aggregate_type == GenericGRU.POOL_HIDDEN:
    in_features = hidden_size*3
  elif aggregate_type == GenericGRU.ATTENTION:
    in_features = hidden_size*s_ngram
  elif aggregate_type == GenericGRU.END_ONLY:
    in_features = hidden_size
  else:
      raise Exception(f'invalid aggregate_type: {aggregate_type}')
  if model_type == GenericGRU.STACKED_GRU_LINEAR:
    in_features *= 5
  elif model_type == GenericGRU.STACKED_GRUS:
    if aggregate_type_gru == GenericGRU.ATTENTION:
      in_features = hidden_size*5
  if kg:
    n_indices = 6
    #if aggregate_type_gru == GenericGRU.POOL_HIDDEN:
    #  input_size = embedding_dim + n_indices
  else:
    n_indices = 0
  input_size = embedding_dim
  weights = torch.tensor([1. - minority_weight, minority_weight])
  s_step = int(np.max([2, np.round(s_ngram*step_scale)]))
  params = Params(save_path=save_path,
                  model_type=model_type,
                  aggregate_type=int(aggregate_type),
                  aggregate_type_gru=int(aggregate_type_gru),
                  headless=headless,
                  add_end_token=add_end_token,
                  oversample=int(oversample),
                  sample_type=int(sample_type),
                  kg=kg,
                  n_indices=n_indices,
                  num_embeddings=num_embeddings,
                  embedding_dim=int(embedding_dim),
                  input_size=int(input_size),
                  hidden_size=int(hidden_size),
                  batch_first=True,
                  in_features=int(in_features),
                  out_features=2,
                  mask_prob=mask_prob,
                  mask_id=mask_id,
                  lr_rate=lr_rate,
                  batch_size=int(batch_size),
                  epochs=int(epochs),
                  weights=weights,
                  shuffle=True,
                  s_ngram=int(s_ngram),
                  s_step=s_step)
  return params


def tensor2array(gru_data):
  """Converts a Tensor to an Array."""
  output_list = []
  for venture in gru_data:
    new_venture = []
    for session in venture:
      if session.shape[0] == 1:
        new_venture.append(np.float64(session[0]))
      else:
        new_venture.append(np.float64(session.cpu().numpy()))
    output_list.append(new_venture)
  return np.array(output_list)


def to_dataloader(x_train: torch.Tensor,
                  y_train: torch.Tensor,
                  batch_size: int,
                  shuffle: bool):
  """Re-maps x, y data into a single DataLoader.
  
  :param x_train: input training data
  :param y_train: input training labels
  :param batch_size: the batching size for the data loader
  :param shuffle: whether or not to shuffle the data
  
  """
  train_data = []
  for i in range(x_train.shape[0]):
    train_data.append([x_train[i], y_train[i]])
  dataloader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=shuffle)
  return dataloader
