from common.tokenizer import Tokenizer

tok = Tokenizer()

text = "<|im_start|>assistant\nHello<|im_end|>"

ids = tok.encode(text)

print(ids)
print(tok.decode(ids))
print(tok.vocab_size)
print(tok.has_special_tokens())