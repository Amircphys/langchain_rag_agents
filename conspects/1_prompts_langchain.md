Короткая ориентация
- LangChain — это glue/инфраструктура вокруг LLM: шаблоны промптов, цепочки, селекторы примеров, память, retrieval и т.д.
- PromptTemplate и ChatPromptTemplate помогают управлять текстом запросов (шаблонами) явно и безопасно.
- FewShotPromptTemplate + ExampleSelector позволяют динамически собирать примеры для few‑shot, контролировать бюджет токенов.
- LengthBasedExampleSelector — полезен, когда нужен отбор примеров по суммарной длине (или по токенам), чтобы не вылезти за контекст LLM.

1) PromptTemplate — что это и почему важно
- Что это: простой шаблон текста с переменными. Позволяет отделить текст промпта от кода, валидировать и переиспользовать.
- Почему: версия промпта должна быть контролируема (версионирование), тестируема, логируемая и подлежат A/B тестированию. PromptTemplate делает это легко.

Пример (Python):
```python
from langchain.prompts import PromptTemplate

template = """Ты — ассистент, помогающий писать деловые e‑mail.
Тон: {tone}
Пиши кратко. Ввод пользователя:
{user_text}
"""
pt = PromptTemplate(input_variables=["tone", "user_text"], template=template)

# Использование
filled = pt.format(tone="дружелюбный", user_text="нужно отписать клиенту о задержке поставки")
print(filled)
```

Советы:
- Делай шаблоны модульными: prefix (системное поведение), suffix (инструкции/формат ответа), и переменные.
- Используй partial variables для неизменяемых частей:
```python
pt = PromptTemplate(
  input_variables=["user_text"],
  partial_variables={"tone": "деловой"},
  template="Тон: {tone}\n{user_text}"
)
```
- Тестируй шаблоны: проверяй, что все input_variables всегда передаются. В CI можно добавить юнит‑тесты.

2) ChatPromptTemplate — «чеканный» формат для чатовых моделей
- Что это: шаблон сообщений в стиле chat (system/human/assistant). В новых чат‑моделях (gpt‑style) лучше использовать ChatPromptTemplate, т.к. конвертирует в список сообщений, а не в один большущий текст.
- Почему: позволяет явно задать system prompt (правила), добавить вспомогательные сообщения ассистента, форматировать подсказки для конкретных ролей.

Пример:
```python
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

system = SystemMessagePromptTemplate.from_template("Ты — эксперт по продуктам. Отвечай кратко.")
human = HumanMessagePromptTemplate.from_template("Опиши фичи: {feature_list}")
chat = ChatPromptTemplate.from_messages([system, human])

# Форматирование:
messages = chat.format_messages(feature_list="поиск, фильтрация, экспорт")
# messages — list of Message objects, можно передать в chat LLM:
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
resp = llm(messages)
```

Полезные приёмы:
- system message — контролирует стиль/ограничения (например, «не придумывай факты»).
- Используй AIMessagePromptTemplate, если хочешь вставить пример поведения ассистента (например, шаблон правильного ответа).
- Для тестов используй format_messages и проверяй результат (структуру и содержание).

3) FewShotPromptTemplate — как правильно давать примеры
- Что это: шаблон, который собирает префикс, несколько примеров (examples), suffix с "новым запросом" и объединяет в итоговый промпт. Удобно для задач классификации/парафраз/инструкций, где few‑shot улучшает результаты.
- Важно: бывает дорогим — примеры увеличивают токены. Тут на сцену выходит ExampleSelector, который выбирает релевантные примеры (semantic) или контролирует длину.

Пример класса для простого few‑shot:
```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "Отзыв: Продукт сломался через день", "output": "негативно"},
    {"input": "Отличный сервис, благодарю", "output": "позитивно"},
]

template_example = "Отзыв: {input}\nКласс: {output}"
fewshot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(input_variables=["input","output"], template=template_example),
    prefix="Ты — классификатор тональности.",
    suffix="Отзыв: {input}\nКласс:",
    example_separator="\n---\n",
    input_variables=["input"]
)

print(fewshot.format(input="Не дождался поддержки"))
```

Как делать правильно:
- Чётко структурируй пример: всегда один формат, одни поля.
- Старайся не использовать слишком много примеров — экономь токены.
- Если у тебя десятки/сотни примеров — используй ExampleSelector (semantic или length‑based).

4) LengthBasedExampleSelector (селектор по длине) — зачем и как его применять
- Проблема: у нас ограниченный контекст (контекстный буфер, max tokens). Мы хотим выбрать набор примеров, который будет релевантен и при этом поместится в бюджет.
- Что делает LengthBasedExampleSelector: хранит pool примеров и выбирает подмножество, суммарная длина (символы/токены) которого ≤ заданного лимита. Ты можешь задавать length_function (например, токены через tiktoken).

Пример использования (с подсчетом токенов через tiktoken):
```python
from langchain.prompts.example_selector import LengthBasedExampleSelector
import tiktoken

# функция длины в токенах
enc = tiktoken.encoding_for_model("gpt-4o-mini")
def tok_len(example: dict) -> int:
    # пример — dict {input, output}; считаем объединённый текст
    text = example["input"] + " " + example.get("output", "")
    return len(enc.encode(text))

examples = [
    {"input":"Отзыв: ...", "output":"позитивно"},
    # много примеров
]

selector = LengthBasedExampleSelector(examples, max_length=800, length_function=tok_len)

# Встраиваем selector в FewShotPromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
example_prompt = PromptTemplate(input_variables=["input","output"], template="{input}\nКласс: {output}")
few = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="Классификатор тональности",
    suffix="Отзыв: {input}\nКласс:",
    input_variables=["input"],
    example_separator="\n\n"
)

# Форматировать промпт: selector выберет набор, который поместится в max_length
prompt_text = few.format(input="Пользователь возмущён качеством")
print(prompt_text)
```

Практические советы по LengthBasedExampleSelector:
- Используй токенный подсчёт (tiktoken) вместо длины в символах — точнее под контекстную квоту LLM.
- Устанавливай max_length с запасом (учитывай и системный префикс, и суффикс, и вход).
- Комбинируй с semantic selector: сначала выбирай наиболее релевантные по семантике, затем ограничивай суммарной длиной.

Дополнительные варианты селекторов
- SemanticSimilarityExampleSelector — использует эмбеддинги и выбирает наиболее похожие примеры. Часто лучше, чем статический length.
- RandomExampleSelector — для A/B тестов.

Практический сценарий: ассистент с RAG + few‑shot
- Pipeline: Retrieval (по векторной БД) → взять top-k контекстов → взять top‑N примеров через semantic selector (или length based) → собрать ChatPromptTemplate: system + examples (в виде assistant/human) + current user → LLM.
- Важно: контекст (retrieved docs) тоже занимает токены — учитывай при выборе примеров.

Production‑оркестр — чеклист
- Контроль бюджета токенов: тестируй реальные промпты и замеряй tokens in/out (в OpenAI/HTTP логах).
- Логирование: всегда логируй промпты и ответы (псевдонимизируй PII), чтобы можно было делать откат/анализ.
- Версионирование промптов: хранить промпты в репозитории, использовать семантические теги (v1, v2).
- Тесты: unit tests на форматирование промптов; интеграционные тесты на LLM при mock/стаб‑ответах.
- Мониторинг качества и стоимости: трекай latency, token usage, ошибочные ответы, hallucinations.
- Безопасность: фильтруй входы на prompt‑injection, добавляй guardrails в system role.
- Кэширование: кешируй часто встречающиеся prompt→answer пары для снижения затрат.
- Отслеживай контекстную «загрязненность»: если FewShot или memory растёт бесконтрольно — результат деградирует.

Типичные подводные камни и как их избежать
- «Примеры слишком большие» → используем LengthBasedExampleSelector.
- «Примеры нерелевантны» → используем semantic selectors и re‑rank.
- «Промпт ломается при нестандартных вводах» → валидируй input_variables, применяй препроцессинг и регекс‑санитизацию.
- «Нельзя отлаживать, т.к. LLM nondeterministic» → для CI используем mocked LLM (deterministic), снэпшоты.
- «Потери при переходе PromptTemplate → ChatPromptTemplate» → учти разницу: chat формирует messages, а обычный — строку; некоторые LLM ожидают именно messages.

Примеры из жизни (чтобы было живо)
1) Email помощник (ChatPromptTemplate + fewshot)
- System: «Ты — помощник, пишущий e‑mail в деловом тоне».
- Примеры (assistant/human парами): 3 коротких шаблонных письма («отписать о задержке», «подтвердить встречу», «отказаться вежливо»).
- Применение: всегда короткие, стандартизированные ответы.

2) Классификация отзывыв (FewShot+LengthSelector)
- У тебя куча аннотированных отзывов. Перед отправкой в LLM берёшь 5-7 наиболее похожих (semantic) но не длиннее 800 токенов (length selector). Это даёт баланс релевантности и бюджета.

3) RAG бот на документации
- Retrieval возвращает кейс‑документы; ChatPromptTemplate включает system + retrieved docs + instruction; иногда добавляем 1–2 пары QA как examples, выбранных LengthBasedExampleSelector, чтобы LLM видел формат ответа.

Полезные функции в LangChain (быстрое перечисление)
- PromptTemplate.format, format_prompt — для отладки.
- ChatPromptTemplate.format_messages — возвращает Message объекты.
- FewShotPromptTemplate с example_selector — динамический подбор.
- ExampleSelector интерфейс — можно кастомно реализовать логику отбора (по времени, по авторитету источника и т.д.).

Мини‑рекомендации по тестам и CI
- Юнит: проверяй, что PromptTemplate.format вызывает ожидаемые вставки.
- Интеграция: mock LLM, assert что при заданных примерах формат промпта совпадает с эталоном.
- A/B: держи несколько версий промптов и замеряй метрики на реальном трафике (CTR/конверсии/accuracy).
