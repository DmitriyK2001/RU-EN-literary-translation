import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import numpy as np
bleu = evaluate.load("bleu")

# Функция для простой токенизации.
# Можно заменить на nltk.word_tokenize(text) при необходимости.
def simple_tokenize(text):
    # Убираем лишние пробелы и переводим в нижний регистр
    text = text.lower().strip()
    # Разбиваем по любому количеству пробельных символов
    tokens = text.split()
    return tokens

# Задайте имена файлов, которые содержат эталонный перевод и перевод модели.
o3mini_score = []
yandex_score = []
qwen_score = []
qwen3b_score = []
qwen3b_clear_score = []
eval_qwen3b = []
eval_qwen3b_clear = []
qwen3b_finetuned_score = []
n = 10
for i in range(1, n + 1):
    file_one = f"{i}_ru.txt"  # эталонный перевод
    file_o3mini = f"{i}_ru o3-mini-high.txt"  # перевод, полученный моделью
    file_yandex = f"{i}_ru Яндекс.txt"
    file_qwen = f"{i}_ru Qwen.txt"
    file_qwen3b = f"{i}_ru_qwen.txt"
    file_qwen3b_clear = f"{i}_ru_qwen_clear.txt"
    file_qwen3b_finetuned = f"{i}_ru Qwen-3B-finetuned.txt"

# Читаем содержимое файлов и объединяем в одну большую строку.
    with open(file_one, 'r', encoding='utf-8') as f:
        reference_text = f.read()
    with open(file_o3mini, 'r', encoding='utf-8') as f:
        o3mini = f.read()
    with open(file_yandex, 'r', encoding='utf-8') as f:
        yandex = f.read()
    with open(file_qwen, 'r', encoding='utf-8') as f:
        qwen = f.read()
    with open(file_qwen3b, 'r', encoding='utf-8') as f:
        qwen3b = f.read()
    with open(file_qwen3b_clear, 'r', encoding='utf-8') as f:
        qwen3b_clear = f.read()
    with open(file_qwen3b_finetuned, 'r', encoding='utf-8') as f:  
        qwen3b_finetuned = f.read()
# Если в файлах несколько строк, они уже образуют одну большую строку.
# Токенизируем тексты:
    reference_tokens = simple_tokenize(reference_text)
    o3mini_tokens = simple_tokenize(o3mini)
    yandex_tokens = simple_tokenize(yandex)
    qwen_tokens = simple_tokenize(qwen)
    qwen3b_tokens = simple_tokenize(qwen3b)
    qwen3b_clear_tokens = simple_tokenize(qwen3b_clear)
    qwen3b_finetuned_tokens = simple_tokenize(qwen3b_finetuned)
# Так как BLEU рассчитывается для набора предложений как эталон,
# обернём reference_tokens в список (т.е. один эталон).
    reference = [reference_tokens]
# Для сглаживания расчёта BLEU (особенно при малом объёме текста) используем SmoothingFunction.
    smoothing = SmoothingFunction().method1

    bleu_score = sentence_bleu(reference, o3mini_tokens, smoothing_function=smoothing)
    o3mini_score.append(bleu_score)
    #print("BLEU-метрика (o3mini):", bleu_score)
    bleu_score = sentence_bleu(reference, yandex_tokens, smoothing_function=smoothing)
    yandex_score.append(bleu_score)
    #print("BLEU-метрика (yandex):", bleu_score)
    bleu_score = sentence_bleu(reference, qwen_tokens, smoothing_function=smoothing)
    qwen_score.append(bleu_score)
    #print("BLEU-метрика (qwen):", bleu_score)
    bleu_score = sentence_bleu(reference, qwen3b_tokens, smoothing_function=smoothing)
    qwen3b_score.append(bleu_score)
    bleu_score = sentence_bleu(reference, qwen3b_clear_tokens, smoothing_function=smoothing)
    qwen3b_clear_score.append(bleu_score)
    bleu_score = sentence_bleu(reference, qwen3b_finetuned_tokens, smoothing_function=smoothing)
    qwen3b_finetuned_score.append(bleu_score)
    predictions = [reference_text,]
    references = [[o3mini],]
    results = bleu.compute(predictions=predictions, references=references)
    references = [[yandex],]
    results = bleu.compute(predictions=predictions, references=references)
    references = [[qwen],]
    results = bleu.compute(predictions=predictions, references=references)
    references = [[qwen3b],]
    results = bleu.compute(predictions=predictions, references=references)
    eval_qwen3b.append(results['bleu'])
    references = [[qwen3b_clear],]
    results = bleu.compute(predictions=predictions, references=references)
    eval_qwen3b_clear.append(results['bleu'])
print(f"o3mini_score:{o3mini_score}", np.mean(np.array(o3mini_score)))
print(f"yandex_score:{yandex_score}", np.mean(np.array(yandex_score)))
#print(f"qwen_score:{qwen_score}", np.mean(np.array(qwen_score)))
print(f"qwen3b_score:{qwen3b_score}", np.mean(np.array(qwen3b_score)))
#print(f"qwen3b_clear_score:{qwen3b_clear_score}", np.mean(np.array(qwen3b_clear_score)))
print(f"qwen3b_finetuned:{qwen3b_finetuned_score}", np.mean(np.array(qwen3b_finetuned_score)))
print(f"eval_qwen3b:{np.mean(np.array(eval_qwen3b))}")
print(f"eval_qwen3b_clear:{np.mean(np.array(eval_qwen3b_clear))}")