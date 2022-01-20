def tokenize_function(examples, tokenizer, max_input_length, max_target_length):
    bos_token = tokenizer.bos_token
    inputs = [bos_token + ' ' + doc for doc in examples['passage']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs