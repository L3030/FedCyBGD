def count_trainable(model,):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (trainable_params / total_params) * 100
    print(
        f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.4f}")
    return trainable_params, total_params, trainable_percentage