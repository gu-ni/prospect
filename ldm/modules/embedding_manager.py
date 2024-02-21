import torch
from torch import nn
import itertools
from ldm.data.personalized import per_img_token_list
from functools import partial
import numpy as np
from ldm.modules.attention import CrossAttention,FeedForward
import PIL
from PIL import Image
import time
DEFAULT_PLACEHOLDER_TOKEN = ["*"]
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
import math
import random

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"
    # return tokens
    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token)


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_per_img=None,
            per_image_tokens=False,
            num_vectors_per_token=2,
            progressive_words=False,
            initializer_words=None,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()
        self.placeholder_embedding = None
        self.embedder=embedder

        self.init = True

        self.cond_stage_model = embedder

        self.progressive_words = progressive_words

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings.token_embedding)
            # get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280
        
        self.get_token_for_string = get_token_for_string
        self.get_embedding_for_tkn = get_embedding_for_tkn
        self.token_dim = token_dim
        self.attention = TransformerBlock(dim=token_dim, n_heads=8, d_head=64, dropout = 0.1, context_dim=token_dim, dim_out = self.max_vectors_per_token*token_dim) # max_vectors_per_token*token_dim -> 10개의 토큰 한번에 계산
        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])
                null_word_token = get_token_for_string('')

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())
                    null_word_embedding = get_embedding_for_tkn(null_word_token.cpu()) ########### null text embedding interpolation ###########

                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(1, 1), requires_grad=True)
                self.initial_embeddings = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(1, 1), requires_grad=False)
                self.null_embeddings = torch.nn.Parameter(null_word_embedding.unsqueeze(0).repeat(1, 1), requires_grad=False) ########### null text embedding interpolation ###########
                
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(1, token_dim), requires_grad=True))
            
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
            self,
            tokenized_text, # tokenized_text: [1, 77]. prompt를 의미
            embedded_text,   # embedded_text: [1, 77, 768]. prompt의 임베딩을 의미
            prospect_words=None,
            timestep=None,
            ca_word=None
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        print('batch',b)

        for placeholder_string, placeholder_token in self.string_to_token_dict.items(): # self.string_to_token_dict.items() = dict_items([('*', tensor(265))])
            if self.initial_embeddings is None:
                print('Working with NO IMGAE mode')
                placeholder_embedding = self.get_embedding_for_tkn('').unsqueeze(0).repeat(self.max_vectors_per_token, 1).to(device)
            else:
                print('Working with IMAGE GUIDING mode')
                
                ca_prob = random.random()
                if ca_word and ca_prob > 2.0:
                    ca_token = self.get_token_for_string(ca_word)
                    with torch.no_grad():
                        ca_embedding = self.get_embedding_for_tkn(ca_token.to(device))
                    cross_attention_word = torch.nn.Parameter(ca_embedding.unsqueeze(0).repeat(1, 1), requires_grad=False)
                    placeholder_embedding = self.attention(self.initial_embeddings.view(b,1,768).to(device), 
                                                           cross_attention_word.view(b,1,768).to(device), 
                                                           timestep)[-1].view(self.max_vectors_per_token,768) # self.attention을 통과시킨 [1, 7680]을 [10, 768]로 view. -> 여기서 self.attention을 거쳐서 임베딩이 반환됨. placeholder_embedding = self.attention 레이어 거친 스페셜 토큰 임베딩들
                else:
                    placeholder_embedding = self.attention(self.initial_embeddings.view(b,1,768).to(device), 
                                                           self.initial_embeddings.view(b,1,768).to(device), 
                                                           timestep)[-1].view(self.max_vectors_per_token,768)
                
            self.placeholder_embedding = placeholder_embedding
            self.placeholder_embeddings=[]
            self.embedded_texts=[]
            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            if self.string_to_param_dict is not None:
                self.string_to_param_dict[placeholder_string] = torch.nn.Parameter(placeholder_embedding, requires_grad=False).to(device) # ParameterDict의 paremeter를 FloatTensor of size 1x768에서 10x768로 바꿈
        
            # plus
            for i in range(self.max_vectors_per_token):
                self.placeholder_embeddings.append(placeholder_embedding[i].view(1,768))
                new_embedded_text = embedded_text.clone().to(device) # 문장 하나 (embedded_text: [1, 77, 768]. prompt에 대한 임베딩)
                new_embedded_text[placeholder_idx] = placeholder_embedding[i].view(1,768).float() # placeholder_idx: 77개 token 중 일부 인덱스. 만약 prompt에 '*'가 있으면 해당 i 개념의 임베딩으로 replace
                self.embedded_texts.append(new_embedded_text) # embedded_texts: max_vectors_per_token * [77, 768]의 (임베딩 수정된) 문장이 저장된 리스트
            """
            #########
            self.placeholder_embeddings.append(placeholder_embedding[i].view(1,768))
            new_embedded_text = embedded_text.clone().to(device) # 문장 하나
            new_embedded_text[placeholder_idx] = placeholder_embedding[i].view(1,768).float() # placeholder_idx: 77개 token 중 일부 인덱스
            self.embedded_texts.append(new_embedded_text)
            ######### (max_vectors_per_token + 1) * [77, 768]
            """
            if prospect_words is not None:
                if isinstance(prospect_words, list) and len(prospect_words)==self.max_vectors_per_token: # 이 두 조건을 만족시키지 않으면 위에 만들어뒀던 self.embedded_texts 리스트 그대로 쓰겠다.
                    print('\nFind word list:',prospect_words)
                    for i in range(len(prospect_words)): # 각 문장에 대해.
                        if isinstance(prospect_words[i],str):
                            words = prospect_words[i].split(' ')
                            if len(words) == 1:
                                if words[0] != '*': # words[0] == '*'이면 위 for 문에서 만들어둔 self.embedded_texts 리스트 그대로 쓰겠다.
                                    none_word_token = self.get_token_for_string(words[0]).to(device)
                                    with torch.no_grad():
                                        new_word_embedding = self.get_embedding_for_tkn(none_word_token).to(device)
                                    new_embeddings = new_word_embedding.unsqueeze(0).view(-1,768).to(device)
                                    new_embedded_text = embedded_text.clone().to(device)   
                                    new_embedded_text[placeholder_idx] = new_embeddings.float()    
                                    self.embedded_texts[i]=new_embedded_text
                            else:
                                #k = i ########## Identity Preservation ##########
                                for j in range(len(words)): # 각 단어에 대해.
                                    if words[j] != '*': # 첫 번째 if-else: 단어가 '*'인지 아닌지
                                        none_word_token = self.get_token_for_string(words[j]).to(device)
                                        with torch.no_grad():
                                            new_word_embedding = self.get_embedding_for_tkn(none_word_token).to(device)
                                    
                                    else:
                                        new_word_embedding = placeholder_embedding[i] # 얘 때문에 prospect_words에서 무조건 마지막 문장이 '*' 없는 문장이어야 함
                                        """
                                        ########## Identity Preservation ##########
                                        new_word_embedding = placeholder_embedding[k]
                                        if k >= 4:
                                            k -= 4
                                        else:
                                            k += 4
                                        ########## Identity Preservation ##########
                                        """
                                    if j == 0: # 두 번째 if-else: 단어가 첫 단어인지 아닌지
                                        new_embeddings = new_word_embedding.unsqueeze(0).view(-1,768).to(device)  
                                    else:
                                        new_embeddings = torch.cat((new_embeddings,new_word_embedding.unsqueeze(0).view(-1,768).to(device)),dim=0) 
                                new_embedded_text = embedded_text.clone().to(device) # new_embedded_text.shape = [1, 77, 768]
                                token_length = new_embeddings.shape[0] # new_embeddings.shape = [(prospect_words 각 문장 length), 768]
                                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                                sorted_rows = placeholder_rows[sort_idx]

                                for idx in range(len(sorted_rows)): # prompt에 '*'가 없으면 이 for 문은 실행 안됨. 그냥 embedded_text(=prompt 임베딩)가 바로 self.embedded_texts[i]에 들어감
                                    row = sorted_rows[idx] # 0
                                    col = sorted_cols[idx] # 1
                                    new_embed_row = torch.cat([new_embedded_text[row][:col], new_embeddings[:token_length], new_embedded_text[row][col + 1:]], axis=0)[:n]
                                    new_embedded_text[row]  = new_embed_row
                                self.embedded_texts[i]=new_embedded_text # embedded_text에 '*'가 없으면 new_embedded_text는 embedded_text(맨처음 prompt의 임베딩)와 동일함. '*'가 있으면 
                        elif isinstance(prospect_words[i],int):
                                new_embedded_text = self.embedded_texts[prospect_words[i]].clone().to(device) 
                                self.embedded_texts[i]=new_embedded_text

        return self.embedded_texts

    def save(self, ckpt_path):
        torch.save({
                    "attention": self.attention,
                    "initial_embeddings": self.initial_embeddings,
                    }, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        if 'attention' in ckpt.keys():
            self.attention = ckpt["attention"]
        else:
            self.attention = None

        if 'initial_embeddings' in ckpt.keys():
            self.initial_embeddings = ckpt["initial_embeddings"]
        else:
            self.initial_embeddings = None

    def embedding_parameters(self):
        return self.attention.parameters()
    
    def embedding_to_coarse_loss(self):        
        pass

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, dim_out=None, max_timestep=1000, timestep_dim=768):
        super().__init__()
        
        ################
        pe = torch.zeros(max_timestep, timestep_dim, requires_grad=False)
        position = torch.arange(max_timestep).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, timestep_dim, 2) * (-math.log(10000.0) / timestep_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # 11/20
        self.dim_out = dim_out
        self.dim = dim
        
        #self.embed_timestep = nn.Linear(timestep_dim, timestep_dim)
        #self.init_attn_embedding = nn.Linear(dim + timestep_dim, dim)
        ################
        
        self.attn1 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff,dim_out=dim_out)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        """
        ################
        self.ff1 = FeedForward(dim, dim_out=dim, mult=2, glu=gated_ff, dropout=dropout)
        self.ff2 = FeedForward(dim, dim_out=dim, mult=2, glu=gated_ff, dropout=dropout)
        self.ff3 = FeedForward(dim, dim_out=dim, mult=2, glu=gated_ff, dropout=dropout)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)
        ################
        """
        
        self.checkpoint = checkpoint

    def forward(self, x, context=None, timestep=None):
        
        # 11/5
        t = self.pe[timestep].unsqueeze(0)
        x = x + t
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.attn2(self.norm2(x), context=context) + x
        #x = self.ff(self.norm3(x))
        # 11/20 ff에도 residual 추가해보기
        x = self.ff(self.norm3(x)) + x.repeat(1, 1, self.dim_out // self.dim)
        
        """
        t = self.pe[timestep].unsqueeze(0)
        t = self.embed_timestep(t)
        x_ = torch.cat([x, t], dim=-1)
        x = self.init_attn_embedding(x_) + x
        
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x))
        """
        
        
        """
        x1 = self.ff1(self.norm3(x))
        x2 = self.ff2(self.norm4(x))
        x3 = self.ff3(self.norm5(x))
        x = torch.cat([x1, x2, x3], dim=1)
        """
        """
        # timestep을 초기 *와 결합하여 추가
        t = self.pe[timestep].unsqueeze(0)
        t = self.embed_timestep(t)
        x_ = torch.cat([x, t], dim=-1)
        x = self.init_attn_embedding(x_) + x
        
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x))
        """
        """
        # timestep을 context에 직접적으로 추가
        x = self.attn1(self.norm1(x), context=context) + x
        t = self.pe[timestep].view(1, 1, 768)
        x = self.attn2(self.norm1(x), context=t) + x
        x = self.ff(self.norm3(x))
        """
        return x
