import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention

class BSARecModel(SequentialRecModel):
    def __init__(self, args):
        super(BSARecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        recommend_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(recommend_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        if getattr(self.args, "ipcm", False):
            loss = loss + self.args.ipcm_lambda * self.ipcm_loss(seq_output, input_ids, answers, neg_answers)

        return loss

    def get_continuation_embedding(self, continuation_ids):
        seq_length = continuation_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=continuation_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(continuation_ids)
        item_embeddings = self.item_embeddings(continuation_ids)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = self.LayerNorm(item_embeddings + position_embeddings)
        return sequence_emb

    def ipcm_loss(self, seq_output, input_ids, answers, neg_answers):
        pos_ids = torch.cat([input_ids[:, 1:], answers.unsqueeze(1)], dim=1)
        hist_mask = (input_ids > 0).float().unsqueeze(-1)
        pos_mask = (pos_ids > 0).float().unsqueeze(-1)

        pred = seq_output * pos_mask
        hist_target = self.get_continuation_embedding(input_ids).detach() * hist_mask
        pos_target = self.get_continuation_embedding(pos_ids).detach() * pos_mask

        window_size = min(getattr(self.args, "ipcm_window_size", 0), pred.size(1))
        if window_size > 0:
            pred = pred[:, -window_size:, :]
            hist_target = hist_target[:, -window_size:, :]
            pos_target = pos_target[:, -window_size:, :]

        pred_freq = torch.fft.rfft(pred, dim=1, norm="ortho")
        hist_freq = torch.fft.rfft(hist_target, dim=1, norm="ortho")
        pos_freq = torch.fft.rfft(pos_target, dim=1, norm="ortho")

        if getattr(self.args, "ipcm_high_freq_only", False):
            cutoff = self.args.c // 2 + 1
            pred_freq = pred_freq[:, cutoff:, :]
            hist_freq = hist_freq[:, cutoff:, :]
            pos_freq = pos_freq[:, cutoff:, :]

        pred_vec = torch.cat([pred_freq.real, pred_freq.imag], dim=-1).flatten(start_dim=1)
        hist_vec = torch.cat([hist_freq.real, hist_freq.imag], dim=-1).flatten(start_dim=1)
        pos_vec = torch.cat([pos_freq.real, pos_freq.imag], dim=-1).flatten(start_dim=1)

        pred_vec = F.normalize(pred_vec, dim=-1)
        hist_vec = F.normalize(hist_vec, dim=-1)
        pos_vec = F.normalize(pos_vec, dim=-1)
        novelty = (1 - torch.sum(hist_vec * pos_vec, dim=-1)).clamp(min=0.0, max=2.0).detach()

        neg_mode = getattr(self.args, "ipcm_neg_mode", "in_batch")
        if neg_mode == "random":
            neg_ids = torch.cat([input_ids[:, 1:], neg_answers.unsqueeze(1)], dim=1)
            neg_target = self.get_continuation_embedding(neg_ids).detach() * pos_mask
            if window_size > 0:
                neg_target = neg_target[:, -window_size:, :]
            neg_freq = torch.fft.rfft(neg_target, dim=1, norm="ortho")
            if getattr(self.args, "ipcm_high_freq_only", False):
                neg_freq = neg_freq[:, cutoff:, :]
            neg_vec = torch.cat([neg_freq.real, neg_freq.imag], dim=-1).flatten(start_dim=1)
            neg_vec = F.normalize(neg_vec, dim=-1)
            pos_logits = torch.sum(pred_vec * pos_vec, dim=-1, keepdim=True)
            neg_logits = torch.sum(pred_vec * neg_vec, dim=-1, keepdim=True)
            logits = torch.cat([pos_logits, neg_logits], dim=1) / self.args.ipcm_tau
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels, reduction="none")
        elif neg_mode == "sfns":
            batch_size, seq_len = input_ids.size()
            num_candidates = getattr(self.args, "ipcm_sfns_candidates", 20)
            candidates = torch.randint(
                1,
                self.args.item_size,
                (batch_size, num_candidates),
                device=input_ids.device,
            )
            answer_expanded = answers.unsqueeze(1).expand_as(candidates)
            replacement = (candidates % (self.args.item_size - 1)) + 1
            candidates = torch.where(candidates == answer_expanded, replacement, candidates)

            neg_prefix = input_ids[:, 1:].unsqueeze(1).expand(-1, num_candidates, -1)
            neg_ids = torch.cat([neg_prefix, candidates.unsqueeze(-1)], dim=-1)
            neg_ids = neg_ids.reshape(batch_size * num_candidates, seq_len)
            neg_mask = pos_mask.unsqueeze(1).expand(-1, num_candidates, -1, -1)
            neg_mask = neg_mask.reshape(batch_size * num_candidates, seq_len, 1)

            neg_target = self.get_continuation_embedding(neg_ids).detach() * neg_mask
            if window_size > 0:
                neg_target = neg_target[:, -window_size:, :]
            neg_freq = torch.fft.rfft(neg_target, dim=1, norm="ortho")
            if getattr(self.args, "ipcm_high_freq_only", False):
                neg_freq = neg_freq[:, cutoff:, :]
            neg_vec = torch.cat([neg_freq.real, neg_freq.imag], dim=-1).flatten(start_dim=1)
            neg_vec = F.normalize(neg_vec, dim=-1)
            neg_vec = neg_vec.view(batch_size, num_candidates, -1)

            candidate_sim = torch.sum(pos_vec.unsqueeze(1) * neg_vec, dim=-1)
            false_negative_threshold = getattr(self.args, "ipcm_sfns_rho", 0.95)
            valid = candidate_sim < false_negative_threshold
            hard_scores = candidate_sim.masked_fill(~valid, -1e4)
            hard_idx = torch.argmax(hard_scores, dim=1)
            fallback_idx = torch.argmin(candidate_sim, dim=1)
            has_valid = valid.any(dim=1)
            selected_idx = torch.where(has_valid, hard_idx, fallback_idx)
            selected_neg_vec = neg_vec[torch.arange(batch_size, device=input_ids.device), selected_idx]

            pos_logits = torch.sum(pred_vec * pos_vec, dim=-1, keepdim=True)
            neg_logits = torch.sum(pred_vec * selected_neg_vec, dim=-1, keepdim=True)
            logits = torch.cat([pos_logits, neg_logits], dim=1) / self.args.ipcm_tau
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels, reduction="none")
        else:
            logits = torch.matmul(pred_vec, pos_vec.transpose(0, 1)) / self.args.ipcm_tau
            labels = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels, reduction="none")

        if getattr(self.args, "ipcm_gate", True):
            loss = loss * novelty

        return loss.mean()

class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
