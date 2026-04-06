import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

print("Загрузка моделей... Это может занять несколько минут.")

language_detector = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')

translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

rus_toxicity_classifier = pipeline('text-classification', model='cointegrated/rubert-tiny-toxicity')
rus_sentiment_classifier = pipeline('sentiment-analysis', model='r1char9/rubert-base-cased-russian-sentiment')

en_toxicity_classifier = pipeline('text-classification', model='martin-ha/toxic-comment-model')
en_sentiment_classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

print("Все рабочие модели успешно загружены!")

SPAM_MARKERS = {"подпис", "канал", "заход", "переход", "профил", "ссылк", "заработ", "крипт"}
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')


def clean_text(text: str) -> str:
    """Очищает текст от эмодзи и большинства символов."""
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def is_kazakh(text: str) -> bool:
    """Проверяет наличие уникальных казахских букв в тексте."""
    KAZAKH_CHARS = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
    return any(char in KAZAKH_CHARS for char in text)

def translate_to_russian(text: str, src_lang_code: str = "kaz_Cyrl") -> str:
    """Переводит текст на русский с помощью модели NLLB."""
    try:
        translator_tokenizer.src_lang = src_lang_code
        encoded_text = translator_tokenizer(text, return_tensors="pt")
        target_lang_id = translator_tokenizer.get_lang_id("rus_Cyrl")
        generated_tokens = translator_model.generate(**encoded_text, forced_bos_token_id=target_lang_id)
        return translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return text

def detect_spam_by_rules(text: str) -> bool:
    """Проверяет текст на наличие явных признаков спама."""
    text_lower = text.lower()
    if URL_PATTERN.search(text_lower):
        return True
    if any(marker in text_lower for marker in SPAM_MARKERS):
        return True
    return False

def get_moderation_verdict(text: str, language: str) -> str:
    """Определяет вердикт модерации: spam, insult или ok."""
    if detect_spam_by_rules(text):
        return 'spam'
    
    toxicity_result = 'non-toxic'
    if language in ['ru', 'kk']: 
        toxicity_result = rus_toxicity_classifier(text)[0]['label']
    elif language == 'en':
        toxicity_result = en_toxicity_classifier(text)[0]['label']
    
    return 'insult' if toxicity_result == 'toxic' else 'ok'



def analyze_comment(comment_text: str) -> dict:
    """
    Полностью анализирует комментарий: очищает, определяет язык, модерирует,
    переводит при необходимости, определяет тональность и тип.
    """
    cleaned_text = clean_text(comment_text)
    if not cleaned_text:
        return {
            'text': comment_text, 'language': 'unknown', 'moderation_verdict': 'ok',
            'sentiment': 'N/A', 'comment_type': 'N/A'
        }

    if is_kazakh(cleaned_text):
        detected_language = 'kk'
    else:
        lang_results = language_detector(cleaned_text, top_k=1)
        detected_language = lang_results[0]['label']
            
    sentiment = "N/A"
    comment_type = "N/A"
    text_to_analyze = cleaned_text
    
    if detected_language == 'kk':
        text_to_analyze = translate_to_russian(cleaned_text, src_lang_code="kaz_Cyrl")

    moderation_verdict = get_moderation_verdict(text_to_analyze, detected_language)

    if moderation_verdict != 'spam':
        if detected_language in ['ru', 'kk']:
            sentiment = rus_sentiment_classifier(text_to_analyze)[0]['label']
        elif detected_language == 'en':
            sentiment = en_sentiment_classifier(text_to_analyze)[0]['label']
        
        descriptive_labels = [
        "the user asks a question about the company's telecommunications service", "the user complains about the company's telecommunications service",
        "the user expresses gratitude to the company's telecommunications service", "the user shares their opinion with the company's telecommunications service"
        ]
        label_map = {
            "the user asks a question about the company's telecommunications service": "question",
            "the user complains about the company's telecommunications service": "complaint",
            "the user expresses gratitude to the company's telecommunications service": "gratitude",
            "he user shares their opinion with the company's telecommunications service": "feedback"
        }
        
        type_result = zero_shot_classifier(text_to_analyze, descriptive_labels)
        top_label = type_result['labels'][0]
        comment_type = label_map.get(top_label, "feedback")

    final_analysis = {
        'text': comment_text,
        'language': detected_language,
        'moderation_verdict': moderation_verdict,
        'sentiment': sentiment,
        'comment_type': comment_type
    }
    return final_analysis

print("\n✅ Аналитический модуль готов к работе!")