from .embedding import PositionalEncoding, Embeddings, Feature_extract
from .mask import generate_lower_triangular_mask, reverse_pad_list, make_pad_mask, add_sos_eos
import torch
from fast_ctc_decode import beam_search, viterbi_search
import torch.nn as nn
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from .encoder import ConformerBlock
from collections import OrderedDict
from pytorch_ranger import Ranger

def ctc_label_smoothing_loss(log_probs, targets, lengths, weights):
    T, N, C = log_probs.shape
    log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
    loss = torch.nn.functional.ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean', zero_infinity=True)
    label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
    return {'loss': loss + label_smoothing_loss, 'ctc_loss': loss, 'label_smooth_loss': label_smoothing_loss}

class Model(torch.nn.Module):
    def __init__(
            self,
            vocab_size = 5,
            dim = 512,
            pad: int = 6,
            sos: int = 5,
            eos: int = 0,
            stride = 10,
            head = 8,
            num_layers = 2,
            weight = 0.5,
            beam_size = 5,
            ffd = 2048
    ):
        super().__init__()   
        self.alphabet = [ "N", "A", "C", "G", "T" ]

        self.feature = Feature_extract(dim, stride)
        self.encoder = nn.ModuleList([ConformerBlock(
            encoder_dim=dim,
            num_attention_heads=head,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=31,
            half_step_residual=True,
            layer_idx=i
        ) for i in range(num_layers*4)])
        self.drop_out = nn.Dropout(0.1)
        self.fc1 = nn.Linear(dim, vocab_size)

        self.emb = Embeddings(dim, vocab_size, pad)
        self.position2 = PositionalEncoding(dim, 0.1, max_len=2000)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=head, batch_first=True, dim_feedforward=ffd)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)      
        self.fc2 = nn.Linear(dim, vocab_size)

        self.embr = Embeddings(dim, vocab_size, pad)
        self.position2r = PositionalEncoding(dim, 0.1, max_len=2000)
        r_decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=head, batch_first=True, dim_feedforward=ffd)
        self.r_decoder = nn.TransformerDecoder(r_decoder_layer, num_layers=num_layers)      
        self.fc2r = nn.Linear(dim, vocab_size)

        self.sos = sos
        self.eos = eos 
        self.vocab_size = vocab_size
        self.pad = pad
        self.stride = stride
        self.weight = weight
        self.beam_size = beam_size

        self.smoothweights = torch.cat([torch.tensor([0.1]), (0.1 / (5 - 1)) * torch.ones(5 - 1)])
        # self.ctc = nn.CTCLoss()
        # self.att = nn.CrossEntropyLoss(ignore_index=pad, label_smoothing=0.1)
        self.att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=pad,
            smoothing=0.1,
            normalize_length=True,
        )

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor = None,
            text: torch.Tensor = None,
            text_lengths: torch.Tensor = None,
    ):
        if text is not None:
            text = text[:, : text_lengths.max()]

        if speech_lengths != None:
            speech = speech[:, :, :speech_lengths.max()]
            encoder_out_lens = torch.ceil(speech_lengths/self.stride).long()
            speech_masks = make_pad_mask(encoder_out_lens).bool().to(speech.device)
        else:
            speech_masks = None
        
        # feature extract
        encoder_in = self.feature(speech)
        
        # Encoder
        for layer in self.encoder:
            if speech_lengths != None:
                encoder_in = layer(encoder_in, 1-speech_masks.int())
            else:
                encoder_in = layer(encoder_in, None)
        encoder_out = self.fc1(self.drop_out(encoder_in))
        ctc_in = nn.functional.log_softmax(encoder_out, -1).permute(1, 0, 2)
        
        if self.training:
            # calc ctc loss
            loss_ctc = ctc_label_smoothing_loss(ctc_in, text, text_lengths, self.smoothweights)
            # loss_ctc = self.ctc(ctc_in, text, encoder_out_lens, text_lengths)
            # Decoder
            # prepare input data for decoder
            text_x = torch.where(text == 0, torch.tensor(6, device=speech.device), text).int().to(speech.device)

            # prepare forward and reverse input
            r_text_x = reverse_pad_list(text_x, text_lengths, float(self.pad))
            ys_in, ys_out = add_sos_eos(text_x, self.sos, self.eos, self.pad)
            r_ys_in, r_ys_out = add_sos_eos(r_text_x, self.sos, self.eos, self.pad)
            ys_in = self.emb(ys_in)
            ys_in = self.position2(ys_in)
            r_ys_in = self.embr(r_ys_in)
            r_ys_in = self.position2r(r_ys_in)
            text_lengths = text_lengths + 1
            tgt_pad_mask = make_pad_mask(text_lengths).bool().to(text.device)
            tgt_mask = generate_lower_triangular_mask(ys_in.shape[1]).bool().to(text.device)

            decoder_out = self.decoder(tgt=ys_in, memory=encoder_in, tgt_mask=~tgt_mask, 
                                       tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=speech_masks)
            decoder_out = self.fc2(decoder_out)

            r_decoder_out = self.r_decoder(tgt=r_ys_in, memory=encoder_in, tgt_mask=~tgt_mask, 
                                           tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=speech_masks)
            r_decoder_out = self.fc2r(r_decoder_out)
            # calc attention loss
            loss_att = (1 - self.weight)*self.att(decoder_out, ys_out.long()) + self.weight*self.att(r_decoder_out, r_ys_out.long())
            loss = self.weight*loss_ctc['loss'] + (1 - self.weight)*loss_att
            return loss
        else:
            # return ctc_in, encoder_out_lens
            return ctc_in
    
    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq
    
def ini_model():
    model = Model()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def load_model(dirname, device='cpu', learning_rate=0.0005, warmup_steps=10000, half=False):
    device = torch.device(device)
    model = Model()
    model.to(device)
    checkpoint = torch.load(dirname, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    optimizer = Ranger(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, 
                                                            factor=0.5, verbose=False, 
                                                            threshold=0.1, min_lr=1e-05)
    scheduler.load_state_dict(checkpoint['scheduler'])
    if half: model = model.half()
    model.train()
    return model, optimizer, scheduler
