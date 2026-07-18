def collect_rank_geometry_fn(
    text_path,
    model_name,
    context_size=128,
    stride=1,
    max_tokens=20000,
    save_path=None,
    topk=2000,
    device="cuda",
    revision=None
):

    model, tokenizer = load_model(model_name, device, revision=revision)
    model = torch.compile(model)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    print(len(tokens))
    tokens = tokens[0:max_tokens]
    print(len(tokens))
    sorted_logits = []

    true_rank = []
    true_prob = []
    entropy = []
    nll = []
    sparse_logits=[]
    gt_token = []
    context_end = []
    vocab_size = model.config.vocab_size
    with torch.no_grad():

        for i in tqdm(range(context_size, len(tokens)-1, stride)):

            context = tokens[i-context_size:i]
            gt = tokens[i]

            input_ids = torch.tensor([context], device=device)

            logits = model(input_ids).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)

            sorted_logits_np = torch.sort(logits, descending=True).values.cpu().numpy()


            sparse_idx = [min(i, vocab_size - 1) for i in [2000, 4999, 9999, vocab_size // 4 - 1, vocab_size // 2 - 1, vocab_size - 1]]
            sparse_logits_np = sorted_logits_np[sparse_idx]
            sorted_logits_np = sorted_logits_np[:topk]

            probs_np = probs.cpu().numpy()
            logits_np = logits.cpu().numpy()

            rank = (
                np.where(
                    np.argsort(logits_np)[::-1] == gt
                )[0][0] + 1
            )

            p_true = probs_np[gt]

            sorted_logits.append(sorted_logits_np)
            sparse_logits.append(sparse_logits_np)
            true_rank.append(rank)
            true_prob.append(p_true)

            entropy.append(
                -(probs_np * np.log(probs_np + 1e-12)).sum()
            )

            nll.append(
                -np.log(p_true + 1e-12)
            )

            gt_token.append(gt)
            context_end.append(i)

    data = {

        "model_name": model_name,
        "text_path": text_path,
        "context_size": context_size,
        "stride": stride,
        "num_tokens": len(tokens),
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,

        "sorted_logits": np.asarray(sorted_logits, dtype=np.float32),
        "sparse_logits": np.asarray(sparse_logits, dtype=np.float32),
        "true_rank": np.asarray(true_rank, dtype=np.int32),
        "true_prob": np.asarray(true_prob, dtype=np.float32),

        "entropy": np.asarray(entropy, dtype=np.float32),
        "nll": np.asarray(nll, dtype=np.float32),

        "gt_token": np.asarray(gt_token, dtype=np.int32),
        "context_end": np.asarray(context_end, dtype=np.int32)
    }

    if save_path is not None:

        np.savez_compressed(
            save_path,
            **data
        )

        print(f"\nSaved to {save_path}")

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    return data
