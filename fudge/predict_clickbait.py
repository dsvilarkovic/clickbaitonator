import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch import Tensor

from data import Dataset
from model import Model
from util import num_params
from constants import *



tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')
classifier_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')


def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)

    article_content = """Australian actor Guy Pearce will return for the iconic soap Neighbours finale on August 1 to reprise his role as Mike Young.
                    Guy, 54, played the troubled Mike from 1986 to 1989, and is now set to make a comeback on the show after 33 years, Metro.co.uk reports.
                    The star's character arcs explored the implications of domestic abuse, student-teacher relationships and dealing with loss of loved ones.
                    Speaking to Metro.co.uk, Guy said: 'It is very exciting and surreal at the same time being back on set again, however it feels like coming home.
                    'It's where it all started for me professionally. I've been asked to come back on occasions over the years and wondered if it was the right thing 
                    to do, but once I knew the show was finishing, I knew I had to do it.'He added that there is 'nothing like being here all together again'
                    , even though he's had a chance to catch-up with other cast members."""

    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0]

    #For loading Clickbait summarizer
    model = AutoModelWithLMHead.from_pretrained(args.model_string, return_dict=True).to(args.device)
    
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    while True:
        results = generate_clickbait(model, 
                        tokenizer, 
                        conditioning_model, 
                        [args.input_text], 
                        dataset_info, 
                        precondition_topk=args.precondition_topk,
                        do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        article_content=article_content,
                        device=args.device)
        # print(results)
        import pdb; pdb.set_trace()


def generate_clickbait(model, 
                        tokenizer, 
                        conditioning_model, 
                        input_text, 
                        dataset_info, 
                        precondition_topk, 
                        length_cutoff, 
                        condition_lambda=1.0, 
                        article_content=None,
                        device='cuda'):
    with torch.no_grad():
        batch_size = len(input_text)
        # encoded_input_article = [tokenizer.encode(article_content, return_tensors='pt',add_special_tokens=False).to(device)] # batch x seq
        max_input_length = 512
        encoded_input_article = tokenizer(article_content, return_tensors='pt',add_special_tokens=False, max_length = max_input_length).to(device) # batch x seq
        # encoded_input_article = torch.cat(encoded_input_article, dim=0)
        # attention_mask = encoded_input_article.new_ones(encoded_input_article.shape).to(device)

        # CHANGE=ko
        encoded_input = tokenizer('<pad>', return_tensors='pt',add_special_tokens=False).to(device) # batch x seq
        # encoded_input = tokenizer('<pad>'+ input_text[0], return_tensors='pt',add_special_tokens=False).to(device) # batch x seq
        # encoded_input = torch.cat(encoded_input, dim=0)
        encoded_input = encoded_input['input_ids']


        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)
        # lengths = 1

        past = None
        use_cache = True

        # CHANGE
        # model_kwargs = {'encoder_outputs': model.get_encoder()(encoded_input_article, attention_mask=attention_mask)}
        model_kwargs = {'encoder_outputs': model.get_encoder()(input_ids=encoded_input_article['input_ids'], 
                                                            attention_mask=encoded_input_article['attention_mask'],
                                                            return_dict=True,
                                                            output_attentions=False,
                                                            output_hidden_states=False),
                        }

        while lengths.max() < length_cutoff:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids = encoded_input_article['input_ids'], 
                decoder_input_ids=encoded_input, 
                # past=past, 
                attention_mask=encoded_input_article['attention_mask'],
                use_cache=use_cache, 
                **model_kwargs
            )

            outputs = model(**model_inputs, return_dict=True)
            logits = outputs.logits[:, -1, :]

            if "past_key_values" in outputs:
                model_kwargs["past"] = outputs.past_key_values

            # logits = model(encoded_input)[0][:, -1, :] # batch x vocab
            top_logits, top_indices = logits.topk(precondition_topk, dim=1) # batch x topk
            new_input_candidates = torch.cat([encoded_input.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
            expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk

            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
            else:
                decoded_outputs = tokenizer.batch_decode(new_input_candidates.view(-1, new_input_candidates.size(-1)), clean_up_tokenization_spaces=False)
                resulting_tokenization = classifier_tokenizer(decoded_outputs, add_special_tokens=False, padding='longest')
                encoded_with_classifier = resulting_tokenization['input_ids']
                attention_mask = torch.tensor(resulting_tokenization['attention_mask']).to(model.device)
                tplus1_candidates_classifier = torch.tensor(encoded_with_classifier).view(batch_size, precondition_topk, -1).to(model.device)

                condition_logits = conditioning_model(tplus1_candidates_classifier.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    None,
                                                    None,
                                                    None,
                                                    attention_mask=attention_mask
                )
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
                condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs

            condition_logits = torch.mean(condition_logits, dim=2)
            full_logits = top_logits + condition_logits * condition_lambda # batch x topk
            post_logits, post_indices = full_logits.topk(precondition_topk, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            # index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
            index_into_top_indices = post_indices[:, torch.multinomial(post_probs, 1).flatten()] # batch

            # next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
            next_indices = top_indices[:, index_into_top_indices] # batch

            # encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
            encoded_input = torch.cat([encoded_input, next_indices.squeeze(1)], dim=1)
            lengths = lengths + 1 # batch

#             print(tokenizer.decode(encoded_input[0], add_special_tokens=False))
        return [tokenizer.decode(s) for s in encoded_input]
    

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')

    parser.add_argument('--in_file', type=str, default=None, required=True, help='text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from text generation at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
