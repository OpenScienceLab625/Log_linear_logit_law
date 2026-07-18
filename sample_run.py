from pathlib import Path
PYTHIA_MODELS   = [ "EleutherAI/pythia-1b"]
CHECKPOINTS     = ["step0", "step1", "step8", "step64", "step512", "step8000"]
SAVE_DIR        = "/content/drive/MyDrive/OrderedLogitGeometry/data"

# one book for checkpoint runs — pick any
DYNAMICS_BOOK   = "/content/drive/MyDrive/OrderedLogitGeometry/gutenberg/Bram Stoker___Dracula.txt"

for model_name in PYTHIA_MODELS:
    short = model_name.split("/")[-1]

    # ── final checkpoint: all books, full K=2000 ─────────────────────────────
    for book in BOOKS:
        bookname = Path(book).stem
        save_path = f"{SAVE_DIR}/{short}_final__{bookname}.npz"
        if Path(save_path).exists():
            print(f"Skipping {save_path}")
            continue
        collect_rank_geometry_fn(
            text_path=book,
            model_name=model_name,
            revision=None,
            topk=2000,
            save_path=save_path,
        )

    for step in CHECKPOINTS:
        for book in BOOKS:
            bookname = Path(book).stem
            save_path = f"{SAVE_DIR}/{short}__{step}__{bookname}.npz"
            if Path(save_path).exists():
                print(f"Skipping {save_path}")
                continue
            collect_rank_geometry_fn(
                text_path=book,
                model_name=model_name,
                revision=step,
                topk=500,
                save_path=save_path,
            )
