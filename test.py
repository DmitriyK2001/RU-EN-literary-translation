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
labels = 'В 1878 году я окончил Лондонский университет, получив звание врача, и сразу же отправился в Нетли, где прошел специальный курс для военных хирургов. После окончания занятий я был назначен ассистентом хирурга в Пятый Нортумберлендский стрелковый полк. В то время полк стоял в Индии, и не успел я до него добраться, как вспыхнула вторая война с Афганистаном. Высадившись в Бомбее, я узнал, что мой полк форсировал перевал и продвинулся далеко в глубь неприятельской территории. Вместе с другими офицерами, попавшими в такое же положение, я пустился вдогонку своему полку; мне удалось благополучно добраться до Кандагара, где я наконец нашел его и тотчас же приступил к своим новым обязанностям.'
predictions = 'В 1878 году я получил степень доктора медицины от Лондонского университета и поступил в Неттли для прохождения курса, который должен был быть выполнен за surgeon в армии. После окончания своих изучений я был присоединен к 5-му Севернскому фузией как ассистент-хирург. Заправка находилась тогда в Индии, и прежде чем я мог участвовать в ней, начался второй фазовый сражение против Афганистана. На посадке в Бомбее я узнал, что корпус его продвигался сквозь горы, и уже был глубоко внутри страны противника. Тем не менее, я следовал вместе с другими офицерами, находящимися в такой же ситуации, и смог безопасно прибыть до Кандахара, где я нашел свой корпус и сразу занялся новыми обязанностями.'
bleu_metric = evaluate.load("bleu")
result = bleu_metric.compute(predictions=predictions, references=[labels])
print(result)