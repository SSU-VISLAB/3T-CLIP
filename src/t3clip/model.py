import open_clip


def create_model(args, device, load_pretrained=True):
    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained
        if load_pretrained and args.pretrained != "none"
        else None,
        device=device,
        force_image_size=(args.img_height, args.img_width),
        force_quick_gelu=args.pretrained == "openai",
        cache_dir=args.cache_dir,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.set_grad_checkpointing(args.grad_checkpointing)
    return model.float(), open_clip.get_tokenizer(args.model_name)
