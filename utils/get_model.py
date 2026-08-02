import segmentation_models_pytorch as smp

UNET_SCALES = {
    'tiny':  (2, (64, 32)),
    'small': (3, (128, 64, 32)),
    'base':  (4, (256, 128, 64, 32)),
    'large': (5, (256, 128, 64, 32, 16)),
}

def get_model(model_name: str, 
              in_c: int = 1, 
              num_classes: int = 1, 
              encoder_weights: str = 'imagenet'):
    """
    Build a segmentation model from a string identifier.

    Naming:
      - UNet:    <scale>_unet_<encoder>
      - FPN:     fpn_<encoder>
      - SegF:    segf_<encoder>
      - PSPNet:  pspn_<encoder>

    UNet scales:
      tiny, small, base, large

    Examples:
      get_model('fpn_resnet18')
      get_model('segf_mit_b3')
      get_model('tiny_unet_resnet34')
    
    Recommended encoder for vessel segmentation: rep_vgg. Variants from tiny to heavy are:
    repvgg_a0, repvgg_a1, repvgg_a2, repvgg_b0, repvgg_b1, repvgg_b1g4, 
    repvgg_b2, repvgg_b2g4, repvgg_b3, repvgg_b3g4, repvgg_d2se
    """

    parts = model_name.split('_')

    # -------------------------
    # UNet: <scale>_unet_<encoder>
    # -------------------------
    if 'unet' in parts:
        if len(parts) < 3:
            raise ValueError("UNet model_name must be '<scale>_unet_<encoder>'")

        scale = parts[0]
        if scale not in UNET_SCALES:
            raise ValueError("Unknown UNet scale '{}'".format(scale))

        encoder_depth, decoder_channels = UNET_SCALES[scale]

        encoder_name = '_'.join(parts[2:])
        if encoder_name.startswith('repvgg'):
            encoder_name = 'tu-' + encoder_name

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_c,
            classes=num_classes,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

    # -------------------------
    # FPN: fpn_<encoder>
    # -------------------------
    elif parts[0] == 'fpn':
        
        encoder_name = '_'.join(parts[1:])
        if encoder_name.startswith('repvgg'):
            encoder_name = 'tu-' + encoder_name

        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_c,
            classes=num_classes,
        )

    # -------------------------
    # SegFormer: segf_<encoder>
    # -------------------------
    elif parts[0] == 'segf':
        
        encoder_name = '_'.join(parts[1:])
        if encoder_name.startswith('repvgg'):
            encoder_name = 'tu-' + encoder_name
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_c,
            classes=num_classes,
        )

    # -------------------------
    # PSPNet: pspn_<encoder>
    # -------------------------
    elif parts[0] == 'pspn':
        encoder_name = '_'.join(parts[1:])
        if encoder_name.startswith('repvgg'):
            encoder_name = 'tu-' + encoder_name
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_c,
            classes=num_classes,
        )

    else:
        raise ValueError("Unknown model_name '{}'".format(model_name))

    setattr(model, 'num_classes', num_classes)
    return model