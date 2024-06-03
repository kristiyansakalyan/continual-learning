import torch
from transformers import Mask2FormerForUniversalSegmentation


class CustomMask2Former(Mask2FormerForUniversalSegmentation):
    def forward(
        self,
        latent_vectors: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        pixel_mask: Tensor,
        **kwargs
    ):

        # Accessing class variables
        config = self.config
        backbone = self.model.pixel_level_module.encoder
        pixel_level_encoder = self.model.pixel_level_module.encoder
        pixel_level_decoder = self.model.pixel_level_module.decoder
        transformer_module = self.model.transformer_module  # Transformer module

        decoder_output = self.decoder(
            backbone_features, output_hidden_states=output_hidden_states
        )

        # Example: Adding custom loss calculation using original loss functions
        original_loss_fn = self.loss

        # For example purposes, let's say we have some dummy targets
        dummy_targets = torch.randint(
            0, 2, (pixel_values.size(0), pixel_values.size(2), pixel_values.size(3))
        )

        # Compute original loss (this is just an example, the actual implementation depends on your task)
        custom_loss = original_loss_fn(outputs.logits, dummy_targets)

        return {
            "logits": outputs.logits,
            "loss": custom_loss,
            "additional_output": torch.randn(
                outputs.logits.size()
            ),  # Dummy additional output
        }
