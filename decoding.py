from PIL import Image
import torch
import torch.nn.functional as F
import requests
from io import BytesIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Note: The parameter "location_features" is used for REG models and
# can be ignored for image captioning.

# helper functions

def img_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image


def img_from_file(path):
    image = Image.open(path).convert('RGB')
    return image


def prepare_img(image, transform, crop_size=224, device=device):
    image = image.resize([crop_size, crop_size], Image.LANCZOS)
    image = transform(image).to(device).view(1, 3, 224, 224)
    return image

# decoding functions

#############################################################################
# 1 likelihood-oriented                                                     #
# (greedy; beam search)                                                     #
#############################################################################


def greedy(model, images, location_features=None, max_len=20, end_id=2):

    # obtain V, v_g
    if location_features is not None:
        # REG model
        V, v_g, captions = model.init_sampler(images, location_features)
    else:
        # captioning model
        V, v_g, captions = model.init_sampler(images)

    sampled_ids = []
    attention = []
    Beta = []
    states = None

    for i in range(max_len):

        scores, states, atten_weights, beta = model.decoder(V, v_g, captions, states)
        predicted = scores.max(2)[1]  # argmax
        captions = predicted

        # Save sampled word, attention map and sentinel at each timestep
        sampled_ids.append(captions)
        attention.append(atten_weights)
        Beta.append(beta)

        # quit if end token has been produced
        if predicted == end_id:
            break

    sampled_ids = torch.cat(sampled_ids, dim=1).squeeze().tolist()
    attention = torch.cat(attention, dim=1).squeeze().tolist()
    Beta = torch.cat(Beta, dim=1).squeeze().tolist()

    return sampled_ids, attention, Beta


def beam_search(model, images,
                location_features=None, beam_width=5, max_len=20):
    ''' Beam search decoder '''

    # obtain V, v_g
    if location_features is not None:
        # REG model
        V, v_g, captions = model.init_sampler(images, location_features)
    else:
        # captioning model
        V, v_g, captions = model.init_sampler(images)

    states = None

    for step in range(max_len):
        scores, states, atten_weights, beta = model.decoder(V, v_g, captions, states)
        t_scores = F.log_softmax(scores, dim=2)  # [1, 1, vocab_len]

        # best k t_scores for each beam (+ indices)
        best_t_scores, token_index = torch.topk(t_scores, beam_width, dim=2)

        best_t_scores = best_t_scores.squeeze()  # best t_scores for each beam
        token_index = token_index.squeeze()  # best tokens for each beam

        if step == 0:
            # expand vectors
            V = V.repeat(beam_width, 1, 1)
            v_g = v_g.repeat(beam_width, 1)
            states = (states[0].repeat(1, beam_width, 1),
                      states[1].repeat(1, beam_width, 1)) # [1,beam_width,512]
            prev_c_scores = best_t_scores.view(-1, 1)  # [beam_width,1]
            captions = token_index.view(-1, 1)  # [beam_width,1]
            candidates = captions

        else:
            non_eos = candidates[:, -1] != 2  # if last token is not <end>
            eos = candidates[:, -1] == 2  # if last token is <end>

            best_t_scores[eos] = -1e20
            token_index[eos] = 2

            # update scores for unfinished candidate captions
            new_c_scores = (
                prev_c_scores.view(-1, 1).expand_as(best_t_scores)[non_eos]
                + best_t_scores[non_eos]  # add probability for best scores
            )

            new_c_scores = new_c_scores.view(-1)  # flatten
            tokens = token_index.view(-1)  # flatten and assign to tokens

            # keep the best beam_width candidates
            new_c_scores, indices = torch.topk(new_c_scores, beam_width, dim=0) # sort scores according to their probability, get top beam_width candidates
            tokens = torch.gather(tokens, 0, indices).view(-1) # select tokens with indices

            idx = (indices/beam_width).long()
            states = (states[0][:, idx, :],
                      states[1][:, idx, :]) # update states - why indices/beam_width

            captions = tokens.view(-1, 1)
            candidates = torch.cat([candidates[idx], captions],
                                   dim=1)

            if candidates[new_c_scores.topk(1)[1]].squeeze().tolist()[-1] == 2:
                return (new_c_scores.topk(1)[0].item(),
                        candidates[new_c_scores.topk(1)[1]].squeeze().tolist())

            prev_c_scores = new_c_scores

    return (new_c_scores.topk(1)[0].item(),
            candidates[new_c_scores.topk(1)[1]].squeeze().tolist())


#############################################################################
# 2 informativeness-oriented                                                #
# (discriminative beam search;                                              #
#  discriminative beam search for multiple distractors)                     #
#############################################################################


def extended_discriminative_greedy(model, img_t, imgs_d,
                                   pos_t=None, pos_d=None,
                                   lambda_=0.5,
                                   max_len=20, end_id=2,
                                   # tracing distractor influence:
                                   return_all_scores=False
                                   ):

    # obtain V_t, v_g_t and V_d, v_g_d
    if pos_t is not None:
        # REG model
        if len(imgs_d) == 0:
            # fallback
            return greedy(
                    model, img_t, location_features=pos_t,
                    max_len=max_len, end_id=end_id
                )
        V_t, v_g_t, captions = model.init_sampler(img_t, pos_t)
    else:
        # captioning model
        V_t, v_g_t, captions = model.init_sampler(img_t)

    states_t = None

    distractors = []
    for i in range(len(imgs_d)):
        # initilizer model to obtain V_d, v_g_d for every distractor image
        if pos_d is not None:
            # REG model
            V_d, v_g_d, captions = model.init_sampler(imgs_d[i], pos_d[i])
        else:
            # captioning model
            V_d, v_g_d, captions = model.init_sampler(imgs_d[i])
        states_d = None
        distractors.append({
            'V_d': V_d,
            'v_g_d': v_g_d,
            'states_d': states_d
        })

    sampled_ids = []
    attention = []
    Beta = []

    all_individual_scores = []

    for step in range(max_len):

        scores_t, states_t, atten_weights, beta = model.decoder(
            V_t, v_g_t, captions, states_t
        )

        individual_scores = [scores_t]

        all_scores = []

        for d in distractors:
            d['scores_d'], d['states_d'], d['atten_weights'], d['beta'] = model.decoder(d['V_d'], d['v_g_d'], captions, d['states_d'])

            individual_scores.append(d['scores_d'])

            # the magic is happening here
            scores = F.log_softmax(scores_t, dim=2) - (1-lambda_)*F.log_softmax(d['scores_d'], dim=2)
            all_scores.append(scores)

        t_scores = sum(all_scores)# / len(all_scores)

        predicted = t_scores.max(2)[1]  # argmax
        captions = predicted

        # Save sampled word, attention map and sentinel at each timestep
        sampled_ids.append(captions)
        attention.append(atten_weights)
        Beta.append(beta)
        all_individual_scores.append(individual_scores)

        # quit if end token has been produced
        if predicted == end_id:
            break

    sampled_ids = torch.cat(sampled_ids, dim=1).squeeze().tolist()
    attention = torch.cat(attention, dim=1).squeeze().tolist()
    Beta = torch.cat(Beta, dim=1).squeeze().tolist()

    if return_all_scores:
        return sampled_ids, attention, Beta, all_individual_scores
    return sampled_ids, attention, Beta


def discriminative_beam_search(model, img_t, img_d,
                               pos_t=None, pos_d=None,
                               lambda_=0.5, beam_width=3, max_len=20):
    '''
    Discriminative beam search decoder
    cf. https://github.com/saiteja-talluri/Context-Aware-Image-Captioning
    '''

    # obtain V_t, v_g_t and V_d, v_g_d
    if pos_t is not None:
        # REG model
        V_t, v_g_t, captions = model.init_sampler(img_t, pos_t)
        V_d, v_g_d, captions = model.init_sampler(img_d, pos_d)
    else:
        # captioning model
        V_t, v_g_t, captions = model.init_sampler(img_t)
        V_d, v_g_d, captions = model.init_sampler(img_d)

    states_t = None
    states_d = None

    for step in range(max_len):

        scores_t, states_t, atten_weights, beta = model.decoder(V_t, v_g_t, captions, states_t)
        scores_d, states_d, atten_weights, beta = model.decoder(V_d, v_g_d, captions, states_d)

        # the magic is happening here
        t_scores = F.log_softmax(scores_t, dim=2) - (1-lambda_)*F.log_softmax(scores_d, dim=2)

        # best k t_scores for each beam (+ indices)
        best_t_scores, token_index = torch.topk(t_scores, beam_width, dim=2)

        best_t_scores = best_t_scores.squeeze()
        token_index = token_index.squeeze()

        if step == 0:
            # expand vectors
            V_t = V_t.repeat(beam_width, 1, 1)
            v_g_t = v_g_t.repeat(beam_width, 1)
            V_d = V_d.repeat(beam_width, 1, 1)
            v_g_d = v_g_d.repeat(beam_width, 1)

            states_t = (states_t[0].repeat(1, beam_width, 1),
                      states_t[1].repeat(1, beam_width, 1)) # [1, beam_width, 512]
            states_d = (states_d[0].repeat(1, beam_width, 1),
                    states_d[1].repeat(1, beam_width, 1)) # [1, beam_width, 512]

            prev_c_scores = best_t_scores.view(-1, 1) # [beam_width, 1]
            captions = token_index.view(-1, 1) # [beam_width, 1]
            candidates = captions

        else:
            non_eos = candidates[:, -1] != 2  # if last token is not <end>
            eos = candidates[:, -1] == 2  # if last token is <end>

            best_t_scores[eos] = -1e20
            token_index[eos] = 2

            # update scores for unfinished candidate captions
            new_c_scores = (
                prev_c_scores.view(-1, 1).expand_as(best_t_scores)[non_eos]
                + best_t_scores[non_eos]  # add probability for best scores
            )

            new_c_scores = new_c_scores.view(-1)  # flatten
            tokens = token_index.view(-1) # flatten and assign to tokens

            # keep the best beam_width candidates
            new_c_scores, indices = torch.topk(new_c_scores, beam_width, dim=0) # sort scores according to their probability, get top beam_width candidates
            tokens = torch.gather(tokens, 0, indices).view(-1) # select tokens with indices

            idx = (indices/beam_width).long()
            # update states_t - select states which correspond to best candidates (divide by beam_width to account for flatten arrays)
            states_t = (states_t[0][:, idx, :],
                      states_t[1][:, idx, :])
            states_d = (states_d[0][:, idx, :],
                      states_d[1][:, idx, :])

            captions = tokens.view(-1, 1)
            candidates = torch.cat([candidates[idx], captions],
                                   dim=1)

            if candidates[new_c_scores.topk(1)[1]].squeeze().tolist()[-1] == 2:
                return (new_c_scores.topk(1)[0].item(),
                        candidates[new_c_scores.topk(1)[1]].squeeze().tolist())

            prev_c_scores = new_c_scores

    return (new_c_scores.topk(1)[0].item(),
            candidates[new_c_scores.topk(1)[1]].squeeze().tolist())


def extended_discriminative_beam_search(model, img_t, imgs_d,
                                        pos_t=None, pos_d=None,
                                        lambda_=0.5, beam_width=3, max_len=20):
    '''
    Discriminative beam search decoder for more than one distractor

    distractor images are handled as a list
    emitter/suppressor method is performed repeatedly for target and every distractor,
    resulting score arrays are then averaged

    cf. https://github.com/saiteja-talluri/Context-Aware-Image-Captioning
    '''

    # obtain V_t, v_g_t and V_d, v_g_d
    if pos_t is not None:
        # REG model
        if len(imgs_d) == 0:
            # fallback
            return beam_search(
                    model, img_t, location_features=pos_t,
                    beam_width=beam_width, max_len=max_len
                )
        V_t, v_g_t, captions = model.init_sampler(img_t, pos_t)
    else:
        # captioning model
        V_t, v_g_t, captions = model.init_sampler(img_t)

    states_t = None

    distractors = []
    for i in range(len(imgs_d)):
        # initilizer model to obtain V_d, v_g_d for every distractor image
        if pos_d is not None:
            # REG model
            V_d, v_g_d, captions = model.init_sampler(imgs_d[i], pos_d[i])
        else:
            # captioning model
            V_d, v_g_d, captions = model.init_sampler(imgs_d[i])
        states_d = None
        distractors.append({
            'V_d': V_d,
            'v_g_d': v_g_d,
            'states_d': states_d
        })

    for step in range(max_len):

        scores_t, states_t, atten_weights, beta = model.decoder(
            V_t, v_g_t, captions, states_t
        )

        all_scores = []

        for d in distractors:
            d['scores_d'], d['states_d'], d['atten_weights'], d['beta'] = model.decoder(d['V_d'], d['v_g_d'], captions, d['states_d'])

            # the magic is happening here
            scores = F.log_softmax(scores_t, dim=2) - (1-lambda_)*F.log_softmax(d['scores_d'], dim=2) # the magic is happening here
            all_scores.append(scores)

        t_scores = sum(all_scores)# / len(all_scores)

        best_t_scores, token_index = torch.topk(t_scores, beam_width, dim=2) # [beam_width, 1, beam_widt] ([1, 1, beam_width] in first step)

        best_t_scores = best_t_scores.squeeze() # [beam_width, beam_width] ([beam_width] in first step)
        token_index = token_index.squeeze() # [beam_width, beam_width] ([beam_width] in first step)

        if step == 0:
            # expand vectors
            V_t = V_t.repeat(beam_width, 1, 1)
            v_g_t = v_g_t.repeat(beam_width, 1)

            states_t = (states_t[0].repeat(1, beam_width, 1),
            states_t[1].repeat(1, beam_width, 1)) # [1, beam_width, 512]

            for d in distractors:
                d['V_d'] = d['V_d'].repeat(beam_width, 1, 1)
                d['v_g_d'] = d['v_g_d'].repeat(beam_width, 1)
                d['states_d'] = (d['states_d'][0].repeat(1, beam_width, 1),
                        d['states_d'][1].repeat(1, beam_width, 1)) # [1, beam_width, 512]

            prev_c_scores = best_t_scores.view(-1, 1) # [beam_width, 1]
            captions = token_index.view(-1, 1) # [beam_width, 1]
            candidates = captions

        else:
            non_eos = candidates[:, -1] != 2  # if last token is not <end>
            eos = candidates[:, -1] == 2  # if last token is <end>

            best_t_scores[eos] = -1e20
            token_index[eos] = 2

            # update scores for unfinished candidate captions
            new_c_scores = (
                prev_c_scores.view(-1, 1).expand_as(best_t_scores)[non_eos]
                + best_t_scores[non_eos]  # add probability for best scores
            )

            new_c_scores = new_c_scores.view(-1)  # flatten
            tokens = token_index.view(-1)  # flatten and assign to tokens

            # keep the best beam_width candidates
            new_c_scores, indices = torch.topk(new_c_scores, beam_width, dim=0) # sort scores according to their probability, get top beam_width candidates
            tokens = torch.gather(tokens, 0, indices).view(-1) # select tokens with indices

            idx = (indices/beam_width).long()
            states_t = (states_t[0][:, idx, :],
                      states_t[1][:, idx, :]) # update states_t - select states which correspond to best candidates (divide by beam_width to account for flatten arrays)

            for d in distractors:
                d['states_d'] = (d['states_d'][0][:, idx, :],
                          d['states_d'][1][:, idx, :])

            captions = tokens.view(-1, 1)
            candidates = torch.cat([candidates[idx], captions],
                                   dim=1)

            if candidates[new_c_scores.topk(1)[1]].squeeze().tolist()[-1] == 2:
                return (new_c_scores.topk(1)[0].item(),
                        candidates[new_c_scores.topk(1)[1]].squeeze().tolist())

            prev_c_scores = new_c_scores

    return (new_c_scores.topk(1)[0].item(),
            candidates[new_c_scores.topk(1)[1]].squeeze().tolist())


#############################################################################
# 4 diversity-oriented                                                      #
# (top-k random sampling; nucleus sampling)                                 #
#############################################################################


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    # https://huggingface.co/transformers/_modules/transformers/generation_logits_process.html

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def top_k_nucleus_sampling(
        model, images, location_features=None,
        top_k=0, top_p=0.0,
        temperature=1.0,
        max_len=20, end_id=2):

    # obtain V, v_g
    if location_features is not None:
        # REG model
        V, v_g, captions = model.init_sampler(images, location_features)
    else:
        # captioning model
        V, v_g, captions = model.init_sampler(images)

    sampled_ids = []
    attention = []
    Beta = []
    states = None

    for i in range(max_len):

        # get logits for next token
        next_token_logits, states, _, _ = model.decoder(
            V, v_g, captions, states)

        # apply temperature coefficient
        next_token_logits = next_token_logits / temperature
        next_token_logits = next_token_logits.squeeze()

        # filter
        filtered_next_token_logits = top_k_top_p_filtering(
            next_token_logits, top_k=top_k, top_p=top_p)

        filtered_next_token_logits = filtered_next_token_logits.unsqueeze(0)

        # sample
        probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        captions = next_token

        # Save sampled token for current time step
        sampled_ids.append(captions)

        # quit if end token has been produced
        if captions == end_id:
            break

    sampled_ids = torch.cat(sampled_ids, dim=1).squeeze().tolist()

    return sampled_ids, attention, Beta


def top_p_sampling(
        model, images, location_features=None,
        top_p=0.0, temperature=1.0,
        max_len=20, end_id=2):
    """
    a.k.a. nucleus decoding
    https://arxiv.org/abs/1904.09751
    """
    return top_k_nucleus_sampling(
        model, images, location_features=location_features,
        top_p=top_p, top_k=0, temperature=temperature,
        max_len=max_len, end_id=end_id)


def top_k_sampling(
        model, images, location_features=None,
        top_k=0, temperature=1.0,
        max_len=20, end_id=2):
    """
    https://arxiv.org/pdf/1805.04833.pdf
    """
    return top_k_nucleus_sampling(
        model, images, location_features=location_features,
        top_k=top_k, top_p=0.0, temperature=temperature,
        max_len=max_len, end_id=end_id)
