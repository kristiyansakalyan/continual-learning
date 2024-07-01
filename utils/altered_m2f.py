from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerForUniversalSegmentationOutput,
    Mask2FormerModelOutput,
)

from utils.common import get_perpixel_features


class M2FWithCLProjHead(Mask2FormerForUniversalSegmentation):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)

        self.dim_in = 1024
        self.proj_dim = 256

        self.sup_con_head = nn.Sequential(
            nn.Conv2d(self.dim_in, self.dim_in, kernel_size=1),
            nn.SyncBatchNorm(self.dim_in),  # TODO: Should we deactivate normalization?
            nn.ReLU(),
            nn.Conv2d(self.dim_in, self.proj_dim, kernel_size=1),
        )

    def forward(self, pixel_values: Tensor, output_hidden_states=None, **kwargs):
        # Regular M2F output
        outputs = super().forward(
            pixel_values, output_hidden_states=output_hidden_states, **kwargs
        )

        if output_hidden_states is not None and output_hidden_states == True:
            # get pixel decoder features
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feats = get_perpixel_features(
                outputs.pixel_decoder_hidden_states,
                outputs.pixel_decoder_last_hidden_state,
                avg=False,
            ).to(device)
            pred_supcon = self.sup_con_head(feats)

            return outputs, pred_supcon
        else:
            return outputs


class CustomMask2Former(Mask2FormerForUniversalSegmentation):
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        pixel_mask: Tensor,
        old_latent_vectors: Optional[Tensor] = None,
        old_mask_labels: Optional[List[Tensor]] = None,
        old_class_labels: Optional[List[Tensor]] = None,
        old_pixel_mask: Optional[Tensor] = None,
        **kwargs
    ):
        output_hidden_states = True
        output_attentions = True
        output_auxiliary_logits = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (
            (old_latent_vectors != None)
            and (old_mask_labels != None)
            and (old_class_labels != None)
            and (old_pixel_mask != None)
        ):

            new_latent_vectors = self.model.pixel_level_module.encoder(
                pixel_values
            ).feature_maps

            backbone_features_1 = torch.cat(
                (new_latent_vectors[0], old_latent_vectors[0]), dim=0
            ).to(device)
            backbone_features_2 = torch.cat(
                (new_latent_vectors[1], old_latent_vectors[1]), dim=0
            ).to(device)
            backbone_features_3 = torch.cat(
                (new_latent_vectors[2], old_latent_vectors[2]), dim=0
            ).to(device)
            backbone_features_4 = torch.cat(
                (new_latent_vectors[3], old_latent_vectors[3]), dim=0
            ).to(device)

            backbone_features = (
                backbone_features_1,
                backbone_features_2,
                backbone_features_3,
                backbone_features_4,
            )

            decoder_output = self.model.pixel_level_module.decoder(
                backbone_features, output_hidden_states=output_hidden_states
            )

            encoder_last_hidden_state = (backbone_features[-1],)
            encoder_hidden_states = (
                tuple(backbone_features) if output_hidden_states else None,
            )
            decoder_last_hidden_state = decoder_output.mask_features
            decoder_hidden_states = decoder_output.multi_scale_features

            concat_pixel_mask = torch.cat((pixel_mask, old_pixel_mask), dim=0).to(
                device
            )
            mask_labels.extend(old_mask_labels)
            class_labels.extend(old_class_labels)

            transformer_module_output = self.model.transformer_module(
                multi_scale_features=decoder_hidden_states,
                mask_features=decoder_last_hidden_state,
                output_hidden_states=True,
                output_attentions=output_attentions,
            )

            pixel_decoder_hidden_states = None
            transformer_decoder_hidden_states = None
            transformer_decoder_intermediate_states = None

            if output_hidden_states:
                pixel_decoder_hidden_states = decoder_hidden_states
                transformer_decoder_hidden_states = (
                    transformer_module_output.hidden_states
                )
                transformer_decoder_intermediate_states = (
                    transformer_module_output.intermediate_hidden_states
                )

            outputs = Mask2FormerModelOutput(
                encoder_last_hidden_state=encoder_last_hidden_state,
                pixel_decoder_last_hidden_state=decoder_last_hidden_state,
                transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
                encoder_hidden_states=encoder_hidden_states,
                pixel_decoder_hidden_states=pixel_decoder_hidden_states,
                transformer_decoder_hidden_states=transformer_decoder_hidden_states,
                transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
                attentions=transformer_module_output.attentions,
                masks_queries_logits=transformer_module_output.masks_queries_logits,
            )

            loss, loss_dict, auxiliary_logits = None, None, None
            class_queries_logits = ()

            for decoder_output in outputs.transformer_decoder_intermediate_states:
                class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
                class_queries_logits += (class_prediction,)

            masks_queries_logits = outputs.masks_queries_logits

            auxiliary_logits = self.get_auxiliary_logits(
                class_queries_logits, masks_queries_logits
            )

            if mask_labels is not None and class_labels is not None:
                loss_dict = self.get_loss_dict(
                    masks_queries_logits=masks_queries_logits[-1],
                    class_queries_logits=class_queries_logits[-1],
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    auxiliary_predictions=auxiliary_logits,
                )
                loss = self.get_loss(loss_dict)

            output_auxiliary_logits = (
                self.config.output_auxiliary_logits
                if output_auxiliary_logits is None
                else output_auxiliary_logits
            )
            if not output_auxiliary_logits:
                auxiliary_logits = None

            output = Mask2FormerForUniversalSegmentationOutput(
                loss=loss,
                class_queries_logits=class_queries_logits[-1],
                masks_queries_logits=masks_queries_logits[-1],
                auxiliary_logits=auxiliary_logits,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
                transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
                encoder_hidden_states=encoder_hidden_states,
                pixel_decoder_hidden_states=pixel_decoder_hidden_states,
                transformer_decoder_hidden_states=transformer_decoder_hidden_states,
                attentions=outputs.attentions,
            )

        else:
            output = super().forward(
                pixel_values, mask_labels, class_labels, pixel_mask
            )

        return output
