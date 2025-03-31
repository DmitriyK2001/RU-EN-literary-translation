from transformers import pipeline
import evaluate

text = 'Mr. Sherlock Holmes, who was usually very late in the mornings, save upon those not infrequent occasions when he was up all night, was seated at the breakfast table. I stood upon the hearth-rug and picked up the stick which our visitor had left behind him the night before. It was a fine, thick piece of wood, bulbous-headed, of the sort which is known as a “Penang lawyer.” Just under the head was a broad silver band nearly an inch across. “To James Mortimer, M.R.C.S., from his friends of the C.C.H.,” was engraved upon it, with the date “1884.” It was just such a stick as the old-fashioned family practitioner used to carry—dignified, solid, and reassuring.“Well, Watson, what do you make of it?”Holmes was sitting with his back to me, and I had given him no sign of my occupation.“How did you know what I was doing? I believe you have eyes in the back of your head.”“I have, at least, a well-polished, silver-plated coffee-pot in front of me,” said he. “But, tell me, Watson, what do you make of our visitor’s stick? Since we have been so unfortunate as to miss him and have no notion of his errand, this accidental souvenir becomes of importance. Let me hear you reconstruct the man by an examination of it.”'
messages = [
    {"role": "user", "content": "Translate to russian:" + text},
]
model_one = "Qwen/Qwen2.5-1.5B-Instruct"
model_two = "./qwen_lora_finetuned_gpu_2"
model = model_two

pipe = pipeline("text-generation", model=model, max_length=512)
predictions = pipe(messages)[0]['generated_text'][1]['content']
print(predictions)