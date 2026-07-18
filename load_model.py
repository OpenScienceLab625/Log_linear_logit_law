def load_model(model_name, device, revision=None):
    global LOADED_MODEL, LOADED_TOKENIZER, LOADED_MODEL_NAME

    kwargs = {"revision": revision} if revision else {}

    cache_key = f"{model_name}__{revision or 'final'}"

    print(f"[load_model] requested: {cache_key}")
    print(f"[load_model] currently cached: {LOADED_MODEL_NAME}")

    if LOADED_MODEL_NAME == cache_key:
        print(f"[load_model] cache HIT — reusing {cache_key}")
        return LOADED_MODEL, LOADED_TOKENIZER

    print(f"[load_model] cache MISS — loading {model_name} revision={revision}")

    model     = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    LOADED_MODEL      = model
    LOADED_TOKENIZER  = tokenizer
    LOADED_MODEL_NAME = cache_key

    print(f"[load_model] loaded and cached as: {cache_key}")

    return model, tokenizer
