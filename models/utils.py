from os.path import join, basename, exists
import hydra
import transformers
import torch.nn.functional as F
import pandas as pd
from collections import Counter
import torch
from torch import nn
from torch.autograd import Function

def compute_emotion_weights(train_csv, emos=None):
    df = pd.read_csv(train_csv)
    if emos is None:
        emos = list(df.columns)[-10:]
    assert len(emos) == 10
    c = Counter(list(df['type']))
    weights = torch.tensor([c[e] for e in emos])
    weights = weights.sum() / weights
    weights = weights / weights.sum()
    return weights

class GradReverse(Function):
    """
        Graident reverse layer
    """
    @staticmethod
    def forward(ctx, x, lambd):
        lambd = torch.tensor(lambd, requires_grad=False)
        ctx.save_for_backward(lambd)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambd, = ctx.saved_tensors
        return (-lambd * grad_output, None)

def grad_reverse(x, lambd=1.0):
    return GradReverse().apply(x, lambd)

class ArgMax(Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, dim=1)
        #output = torch.zeros_like(input, device=input.device)
        #output.scatter_(1, idx, 1)
        return idx
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def sf_argmax(x):
    return ArgMax().apply(x)

def linear_lr_with_warmup(current_step, warmup_steps, flat_steps, training_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    elif current_step < (warmup_steps + flat_steps):
        return 1.0
    return max(
        0.0, float(training_steps - current_step) / float(max(1, training_steps - (warmup_steps + flat_steps)))
    )

def load_ssl_model(model_path):
    ocwd = hydra.utils.get_original_cwd()
    path = join(ocwd, model_path)
    config = {'layerdrop': 0, 'gradient_checkpointing': False}
    if 'wav2vec2' in model_path:
        model = transformers.Wav2Vec2ForPreTraining.from_pretrained(model_path, **config)
    elif 'hubert' in model_path:
        model = transformers.HubertModel.from_pretrained(model_path, **config)
    return model

def get_contrast_loss(similarity):
    '''
    L(e_ji) = 1-sigmoid(S_jij)+max_k(sigmoid(S_jik))
    :param similarity: shape -> (N, M, N)
    :return:
    loss = sum_ji(L(e_ji))
    '''

    # some inplace operation
    # one of the variables needed for gradient computation has been modified by an inplace operation
    # so I choose to implement myself
    sigmoid = 1 / (1 + torch.exp(-similarity))
    same_index = list(range(similarity.shape[0]))
    loss_1 = torch.mean(1-sigmoid[same_index, :, same_index])
    sigmoid[same_index, :, same_index] = 0
    loss_2 = torch.mean(torch.max(sigmoid, dim=2)[0])

    loss = loss_1 + loss_2
    return loss

def get_softmax_loss(similarity):
    '''
    L(e_ji) = -S_jij) + log(sum_k(exp(S_jik))
    :param similarity: shape -> (N, M, N)
    :return:
    loss = sum_ji(L(e_ji))
    '''
    same_index = list(range(similarity.shape[0]))
    loss = torch.mean(torch.log(torch.sum(torch.exp(similarity), dim=2) + 1e-6)) - torch.mean(similarity[same_index, :, same_index])
    return loss

def get_similarity(embedding):
    '''
    get similarity for input embedding
    :param embedding: shape -> (N, M, feature)
    :return:
    similarity: shape -> (N, M, N)
    '''
    embedding_mean_include = calculate_centroid_include_self(embedding)
    embedding_mean_exclude = calculate_centroid_exclude_self(embedding)

    similarity = calculate_similarity(embedding, embedding_mean_include) # shape (N, M, N)
    similarity_j_equal_k = calculate_similarity_j_equal_k(embedding, embedding_mean_exclude) # shape (N, M)
    similarity = combine_similarity(similarity, similarity_j_equal_k)
    return similarity

def calculate_centroid_include_self(embedding):
    '''
    calculate centroid embedding. For each embedding, include itself inside the calculation.
    :param embedding: shape -> (N, M, feature_dim)
    :return:
    embedding_mean: shape -> (M, feature_dim)
    '''
    N, M, feature_dim = embedding.shape
    embedding_mean = torch.mean(embedding, dim=1)
    return embedding_mean

def calculate_centroid_exclude_self(embedding):
    '''
    calculate centroid embedding. For each embedding, exclude itself inside the calculation.
    :param embedding: shape -> (N, M, feature_dim)
    :return:
    embedding_mean: shape -> (N, M, feature_dim)
    '''
    N, M, feature_dim = embedding.shape
    embedding_sum = torch.sum(embedding, dim=1, keepdim=True) # shape -> (N, 1, feature_dim)
    embedding_mean = (embedding_sum - embedding) / (M-1)
    return embedding_mean

def calculate_similarity(embedding, centroid_embedding):
    '''
    calculate similarity S_jik
    :param embedding: shape -> (N, M, feature_dim)
    :param centroid_embedding: -> (N, feature_dim)
    :return:
    similarity: shape -> (N, M, N)
    '''
    N, M, feature_dim = embedding.shape
    N_c, feature_dim_c = centroid_embedding.shape
    assert N == N_c and feature_dim == feature_dim_c, "dimension wrong in get_similarity_include_self!"

    centroid_embedding = centroid_embedding.unsqueeze(0).unsqueeze(0).expand(N, M, -1, -1)
    assert centroid_embedding.shape == (N, M, N, feature_dim), "centroid embedding has wrong expansion in get_similarity_include_self."
    embedding = embedding.unsqueeze(2)
    similarity = F.cosine_similarity(embedding, centroid_embedding, dim=3)
    return similarity

def calculate_similarity_j_equal_k(embedding, centroid_embedding):
    '''
    calculate cimilarity S_jik for j == k
    :param embedding: shape -> (N, M, feature)
    :param centroid_embedding: shape -> (N, M, feature)
    :return:
    similarity: shape -> (N, M)
    '''
    N, M, feature_dim = embedding.shape
    N_c, M_c, feature_dim_c = centroid_embedding.shape
    assert N==N_c and M==M_c and feature_dim==feature_dim_c, "dimension wrong in get_similarity_exclude_self!"

    similarity = F.cosine_similarity(embedding, centroid_embedding, dim=2)
    return similarity

def combine_similarity(similarity, similarity_j_equal_k):
    same_index = list(range(similarity.shape[0]))
    similarity[same_index, :, same_index] = similarity_j_equal_k[same_index, :]
    return similarity
