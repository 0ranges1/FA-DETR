import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules import MSDeformAttn
from models.attention import SingleHeadSiameseAttention

from util.misc import inverse_sigmoid


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layers = nn.ModuleList()
        FIM_position = 0 # FIM's location (at the first encoder layer)
        for i in range(num_encoder_layers):
            is_before_FIM = i < FIM_position
            encoder_layers.append(
                DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                  dropout, activation,
                                                  num_feature_levels, nhead, enc_n_points, QSAttn=(i == FIM_position), is_before_FIM=is_before_FIM)
            )
        self.fa_encoder = DeformableTransformerEncoder(encoder_layers, num_encoder_layers)


        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)

        self.fa_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=return_intermediate_dec)

        if self.num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        if self.num_feature_levels > 1:
            normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, class_prototypes, tsp):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape 
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # （bs, c, hw）
            mask = mask.flatten(1) # (_, hxw)
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # （bs, c, hw）
            if self.num_feature_levels > 1:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # ********************  Encoder  ********************
        memory, query_feature = self.fa_encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, class_prototypes, tsp
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid() # （batchsize, num_object_query, d_model） -> （batchsize, num_object_query, 2)
        init_reference_out = reference_points

        # ********************  Decoder  ********************    
        hs, inter_references = self.fa_decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten
        )

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, query_feature
    


    def forward_supp_branch(self, srcs, masks, pos_embeds, support_boxes): 

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        class_prototypes, feat_before_RoIAlign_GAP = self.fa_encoder.forward_supp_branch(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, support_boxes
        )

        # class_prototypes = （episode_size, d_model）
        return class_prototypes, feat_before_RoIAlign_GAP



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, QSAttn=False, is_before_FIM=True):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.QSAttn = QSAttn
        self.is_before_FIM = is_before_FIM

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

       
        # siamese attention
        if self.QSAttn:  
            self.siamese_attn = SingleHeadSiameseAttention(d_model)
            self.linear_weight = nn.Linear(d_model, 1)

        # ffn  
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        

    @staticmethod
    def with_pos_embed(tensor, pos): 
        return tensor if pos is None else tensor + pos
    

    def forward_ffn(self, src): 
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        src = self.norm3(src)
        return src


    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask, class_prototypes, tsp):
        src_sa = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src_sa) 
        src = self.norm1(src)

        query_feature = None
        if self.QSAttn: 
            query_feature = src
          
            query_img_h, query_img_w = spatial_shapes[0, 0], spatial_shapes[0, 1] 
            query_feature = query_feature.transpose(1, 2).reshape(query_feature.shape[0], -1, query_img_h, query_img_w) # (batchsize, c, h, w)
            
            src = self.siamese_attn(src, class_prototypes, tsp)
        
        src = self.forward_ffn(src)

        return src, query_feature


    def forward_supp_branch(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask, support_boxes):
        src_sa = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src_sa)
        src = self.norm1(src) 
        
        feat_before_RoIAlign_GAP = src


        support_img_h, support_img_w = spatial_shapes[0, 0], spatial_shapes[0, 1]
        

        # ===================================================================================
        # averaged prototypes (GAP) ⬇️
        # class_prototypes = torchvision.ops.roi_align(
        #     src.transpose(1, 2).reshape(src.shape[0], -1, support_img_h, support_img_w), # N, C, support_img_h, support_img_w
        #     support_boxes,
        #     output_size=(7, 7),
        #     spatial_scale=1 / 32.0, 
        #     aligned=True).mean(3).mean(2) 
        

        # Weighted Prototypes ⬇️
        class_prototypes = torchvision.ops.roi_align(
            src.transpose(1, 2).reshape(src.shape[0], -1, support_img_h, support_img_w), # N, C, support_img_h, support_img_w
            support_boxes,
            output_size=(7, 7),
            spatial_scale=1 / 32.0, 
            aligned=True) 
        # ===================================================================================
        
        class_prototypes = class_prototypes.view(src.shape[0], src.shape[2], -1).transpose(1, 2)
        weight = self.linear_weight(class_prototypes)
        weight = weight.softmax(1) # N, HW, 1
        class_prototypes = (weight * class_prototypes).sum(1) # N, C
        
        # ffn
        src = self.forward_ffn(src)
        
        return src, class_prototypes, feat_before_RoIAlign_GAP




class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layers, num_layers):
        super().__init__()
        self.layers = encoder_layers
        self.num_layers = num_layers
        assert self.num_layers == len(self.layers)
    

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] 
        return reference_points 

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask, class_prototypes, tsp):
        output = src # b, hw, c

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        for i, layer in enumerate(self.layers):
            if layer.QSAttn == True:
                output, query_feature = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, class_prototypes, tsp)
            else:
                output, _ = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, class_prototypes, tsp)

        return output, query_feature

    def forward_supp_branch(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask, support_boxes):
        output = src

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        for i, layer in enumerate(self.layers):
            if layer.is_before_FIM or layer.QSAttn:
                output, class_prototypes, feat_before_RoIAlign_GAP = layer.forward_supp_branch(
                    output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, support_boxes
                )
        return class_prototypes, feat_before_RoIAlign_GAP # class_prototypes: (episode_size, d_model)， feat_before_RoIAlign_GAP: (episode_size, hw, d_model)



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_ffn

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # ==============================================================================
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(self.d_model)
        # ==============================================================================
        

        # ffn
        self.linear1 = nn.Linear(self.d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(d_ffn, self.d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(self.d_model)
        


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt # (batchsize, num_queries, self.d_model)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement
        self.bbox_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos, src_padding_mask):
        output = tgt 

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None] 
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate: 
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points 




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu/glu/leaky_relu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
    )
