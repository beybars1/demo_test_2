from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode the input text
input_text = "I enjoy walking with my cute dog"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=100, temperature=0.8, do_sample=True)

# Decode the output
output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

print(output_text)