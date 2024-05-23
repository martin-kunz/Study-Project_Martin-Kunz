from data_preprocessing import replace_tags_with_mask

tags_to_replace = [
    "I-ORG",
    "B-LOC",
    "I-PER",
    "I-LOC",
    "I-MISC",
]

replace_tags_with_mask("data/conll03_test.txt", "data/masked_conll03_test.txt", tags_to_replace)
replace_tags_with_mask("data/conll03_train.txt", "data/masked_conll03_train.txt", tags_to_replace)
replace_tags_with_mask("data/conll03_valid.txt", "data/masked_conll03_valid.txt", tags_to_replace)
