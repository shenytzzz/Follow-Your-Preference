#################################################################################################
# Modified from https://github.com/nftblackmagic/catvton-flux/blob/main/src/flux/train_utils.py #
#################################################################################################
"""Utility functions for preparing images, masks and latents for training.

Only for Flux DPO training scripts (packing/unpacking
latents, constructing conditioning tensors, mask transformations).
"""

import torch
from einops import rearrange


def prepare_image_with_mask(
    image_processor,
    mask_processor,
    vae,
    vae_scale_factor,
    image,
    mask,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    is_cloth=False,
):
    # Prepare image 
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    # print("image.shape", image.shape)
    image_batch_size = image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)

    # Prepare mask
    if isinstance(mask, torch.Tensor):
        pass
    else:
        mask = mask_processor.preprocess(mask, height=height, width=width)
    mask = mask.repeat_interleave(repeat_by, dim=0)
    mask = mask.to(device=device, dtype=dtype)

    # Get masked image
    masked_image = image.clone()
    masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1

    # Encode to latents
    image_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    # print("image_latents.shape", image_latents.shape)
    mask = torch.nn.functional.interpolate(
        mask, size=(height // vae_scale_factor * 2, width // vae_scale_factor * 2)
    )
    if is_cloth:
        mask = mask
    else:
        mask = 1 - mask

    control_image = torch.cat([image_latents, mask], dim=1)

    # Pack cond latents
    packed_control_image = pack_latents(
        control_image,
        batch_size * num_images_per_prompt,
        control_image.shape[1],
        control_image.shape[2],
        control_image.shape[3],
    )

    return packed_control_image, height, width


def prepare_fill_with_mask(
    image_processor,
    mask_processor,
    vae,
    vae_scale_factor,
    masked_image,
    mask,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
):
    """
    Prepares image and mask for fill operation with proper rearrangement.
    Focuses only on image and mask processing.
    """
    # Determine effective batch size
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    # Prepare image
    if isinstance(masked_image, torch.Tensor):
        pass
    else:
        masked_image = image_processor.preprocess(
            masked_image, height=height, width=width
        )

    masked_image = masked_image.to(device=device, dtype=dtype)

    # Prepare mask with specific processing
    if isinstance(mask, torch.Tensor):
        pass
    else:
        mask = mask_processor.preprocess(mask, height=height, width=width)

    mask = mask.to(device=device, dtype=dtype)

    # Encode to latents
    image_latents = vae.encode(masked_image).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)
    image_latents = pack_latents(
        image_latents,
        batch_size,
        num_channels_latents=vae.config.latent_channels,
        height=height,
        width=width,
    )

    # Process mask following the example's specific rearrangement
    mask = mask[
        :, 0, :, :
    ]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
    mask = mask.view(
        batch_size, height, vae_scale_factor, width, vae_scale_factor
    )  # batch_size, height, 8, width, 8
    mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    mask = mask.reshape(
        batch_size, vae_scale_factor * vae_scale_factor, height, width
    )  # batch_size, 8*8, height, width
    mask = pack_latents(
        mask,
        batch_size,
        vae_scale_factor * vae_scale_factor,
        height,
        width,
    )
    mask = mask.to(torch.bfloat16)
    # Combine image and mask
    image_cond = torch.cat([image_latents, mask], dim=-1)

    return image_cond, height, width


def prepare_image_with_mask_sd3(
    image_processor,
    mask_processor,
    vae,
    vae_scale_factor,
    image,
    mask,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    is_cloth=False,
):
    # Prepare image
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    # print("image.shape", image.shape)
    image_batch_size = image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)

    # Prepare mask
    if isinstance(mask, torch.Tensor):
        pass
    else:
        mask = mask_processor.preprocess(mask, height=height, width=width)
    mask = mask.repeat_interleave(repeat_by, dim=0)
    mask = mask.to(device=device, dtype=dtype)

    # Get masked image
    masked_image = image.clone()
    masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1

    # Encode to latents
    image_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    # print("image_latents.shape", image_latents.shape)
    mask = torch.nn.functional.interpolate(
        mask, size=(height // vae_scale_factor, width // vae_scale_factor)
    )
    if is_cloth:
        mask = mask
    else:
        mask = 1 - mask

    control_image = torch.cat([image_latents, mask], dim=1)

    return control_image, height, width


def prepare_image_for_refnet(
    image_processor,
    vae,
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
):
    # Prepare image
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    # print("image.shape", image.shape)
    image_batch_size = image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)

    # Encode to latents
    image_latents = vae.encode(image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    # Pack cond latents
    packed_image = pack_latents(
        image_latents,
        batch_size * num_images_per_prompt,
        image_latents.shape[1],
        image_latents.shape[2],
        image_latents.shape[3],
    )

    return packed_image, height, width


def prepare_image_for_refnet_sd3(
    image_processor,
    vae,
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance=False,
    guess_mode=False,
):
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    image_batch_size = image.shape[0]

    # Prepare image
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    # Encode to latents
    # print("masked_image.dtype", masked_image.dtype)
    image_latents = vae.encode(image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    return image_latents


# Copied from diffusers.pipelines.flux.pipeline_flux._pack_latents
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_latents(
    vae_scale_factor,
    batch_size,
    height,
    width,
    dtype,
    device,
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latent_image_ids = prepare_latent_image_ids(
        batch_size, height // 2, width // 2, device, dtype
    )
    return latent_image_ids


def decode_packed_image(
    packed_control_image,
    vae,
    vae_scale_factor,
    height,
    width,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
):
    # Unpack latents
    control_image = unpack_latents(
        packed_control_image,
        batch_size * num_images_per_prompt,
        5,  # 4 channels for image_latents + 1 for mask
        height // vae_scale_factor * 2,
        width // vae_scale_factor * 2,
    )

    # Split control_image into image_latents and mask
    image_latents, mask = torch.split(control_image, [4, 1], dim=1)

    # Decode latents
    image_latents = image_latents / vae.config.scaling_factor + vae.config.shift_factor
    image = vae.decode(image_latents.to(vae.dtype)).sample

    # Interpolate mask back to original size
    mask = torch.nn.functional.interpolate(mask, size=(height, width))
    mask = 1 - mask  # Invert mask

    # Apply mask to image
    masked_image = image.clone()
    masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1

    return image, masked_image, mask


# Helper function to unpack latents
def unpack_latents(packed_latents, batch_size, num_channels, height, width):
    unpacked = packed_latents.reshape(
        batch_size, height // 2, width // 2, num_channels, 2, 2
    )
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    unpacked = unpacked.reshape(batch_size, num_channels, height, width)
    return unpacked


def get_image_proj(
    transformer,
    image_prompt: torch.Tensor,
    device,
):
    if (
        transformer.auto_processor is not None
        and transformer.image_encoder is not None
        and transformer.garment_adapter_improj is not None
    ):
        # encode image-prompt embeds
        # transformer.image_encoder.to(device=device, dtype=torch.float32)
        # print("image_prompt.dtype", image_prompt.dtype)
        image_prompt = transformer.clip_image_processor(
            images=image_prompt, return_tensors="pt"
        ).pixel_values

        image_prompt = image_prompt.to(device)
        image_prompt_embeds = transformer.image_encoder(image_prompt).image_embeds.to(
            device=device,
            dtype=torch.bfloat16,
        )

        # encode image
        # print("image_prompt_embeds.shape", image_prompt_embeds.shape)
        image_proj = transformer.garment_adapter_improj(image_prompt_embeds)

        return image_proj
    else:
        print("No image projector found")
        return None


def encode_images_to_latents(
    vae, pixel_values, weight_dtype, height, width, image_processor=None
):

    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    return model_input


@staticmethod
def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents
