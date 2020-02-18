from transformers import BertTokenizer, pipeline, BertModel, BertConfig

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


config = BertConfig.from_json_file('tfrecords/config.json')
model = BertModel.from_pretrained(
    'tfrecords/model.ckpt-10000.index', from_tf=True, config=config)
# tokenizer = BertTokenizer.from_pretrained('tfrecords/')


fill_mask = pipeline(
    "sentiment-analysis",
    model=model,
    # tokenizer=tokenizer
)


result = fill_mask("मैले भात <MASK>")
print(fill_mask)
