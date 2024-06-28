from data_preprocessing import replace_tags_with_mask

tags = ["I-PER", "B-PER", "I-LOC", "B-LOC", "I-ORG", "B-ORG", "I-MISC", "B-MISC", "O"]
subsets = ["test", "train", "valid"]

for subset in subsets:
    for tag in tags:
        replace_tags_with_mask(
            file_path=f"data/conll03/{subset}/{subset}.txt",
            tag=tag,
            output_path=f"data/conll03/{subset}/{tag}_masked.txt",
            mask_token="[MASK]",
        )
