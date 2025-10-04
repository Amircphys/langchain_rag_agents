📖 Память позволяет LLM помнить предыдущие взаимодействия с пользователем. По умолчанию LLM является `stateless`, что означает независимую обработку каждого входящего запроса от других взаимодействий. На самом деле, когда мы вводим запросы в интерфейсе ChatGPT, под капотом работает память - т.е. модель отвечает нам, исходя из контекста всего диалога. 

🧠 `InMemoryChatMessageHistory` - храним историю в памяти

📖 Один из способов сохранять историю диалога в `LangChain` - это хранение контеста в оперативной памяти (`InMemoryChatMessageHistory`), подходит, например для сценариев, когда всё взаимодействие с пользователем происходит за один сеанс (в одном диалоге), после перезапуска всё обнулится.

Чтобы прикрутить к любому `Runnable` объекту память, потребуется обёртка `RunnableWithMessageHistory`. Объект `Runnable` - наиболее близок по смыслу к "запускаемый объект".

Чтобы в памяти не путались диалоги от разных пользователей и по различным темам при запросе  в конфигурацию добавляется `sessiоn ID`. Можно добавить и несколько ключей, например ещё `user ID`, чтобы отельно идентифицировать несколько различных диалогов на разные темы с одним пользователем (вспомните как создаёте новый чат с ChatGPT).


```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key: str = os.getenv("api_key", "")
base_url: str = os.getenv("base_url", "")

llm = ChatOpenAI(
    model="gpt-5-nano-2025-08-07",
    api_key=api_key,
    base_url=base_url,
)

# Создаём конфиг для нашего Runnable, чтобы указать ему session_id при вызове
config = {"configurable": {"session_id": "1"}}

llm_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history
)

# к вызову добавляем параметр config
llm_with_history.invoke("Привет, ChatGPT! Меня зовут Иван. Как дела?", config=config).content
print(store['1'])
llm_with_history.invoke("Сможешь помочь мне в написании кода на Python?", config=config).content
print(store['1'])
```

<div class="alert alert-success" style="background-color:#e6e6; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
    
**Видно, что благодаря запоминанию контекста, модель знала, что нам нужен ответ именно c синтаксисом языка Python.**

В `InMemoryChatMessageHistory` сообщения сохраняются парами `HumanMessage` - `AIMessage`.  </div>

```python
get_session_history('1').clear()
```
<div style="background-color:#e6e6; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

Модель не смогла ответить, так как очистили предыдущий диалог. 

**Контекст (историю) можно задавать вручную, подгружать из файла или БД.** </div>


```python
from langchain_core.messages import HumanMessage, AIMessage
get_session_history('1').clear()

# Допустим эти примеры мы взяли из базы данных
user_message = HumanMessage(content="Привет меня зовут Алерон?")
ai_message = AIMessage(content="Привет, Алерон. Чем могу помочь?")
# загрузим диалог в память методом add_messages, принимающим список сообщений
get_session_history('1').add_messages([user_message, ai_message])

store
```

<div class="alert alert-info" style="padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
    
📖 **Для подгрузки в память так же можно использовать следующие методы:**
* `add_message` - добавляет любой тип сообщения
* `add_ai_message` - добавляет сообщение типа `AIMessage`
* `add_user_message` - добавляет сообщение типа `HumanMessage`
* `aadd_messages` - асинхронное добавление сообщений </div>


# <center id="d3"> `ChatPromptTemplate` + память = цепочка всё помнит 🧠 </center>

До этого мы вызывали только LLM, теперь давайте разберёмся как добавить память к простой цепочке, в которой есть шаблон промпта.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Создадим шаблон промпта для ассистента
# Оставим в шаблоне "заглушку" - MessagesPlaceholder, в которую будет подставляться история диалога
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты полезный ассистент, отлично разбирающийся в {topic}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# собираем простую цепочку
chain = prompt | llm

# добавляем сохранение истории взаимодействий
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history, 
    input_messages_key="question",  # указываем название переменой запроса
    history_messages_key="chat_history", # название переменной для истории из шаблона
)

chain_with_history.invoke(
    {"topic": "математика", "question": "Чему равен синус 30?"},
    config={"configurable": {"session_id": "2"}}).content

chain_with_history.invoke(
    {"topic": "математика", "question": "А чему равен косинус?"},
    config={"configurable": {"session_id": "2"}}).content
```

<div class="alert alert-success" style="background-color:#e6e6; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
    
Из контекста модель поняла, что мы спрашиваем про косинус угла 30 градусов.

👀 Посмотрим на `store` - теперь по 2 разным ключам хранятся истории разных диалогов.</div>


<div class="alert alert-info" style="padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
    
📖 **Недостатки `InMemoryChatMessageHistory`:**
* История хранится в оперативной памяти. При интенсивных диалогах и большом количестве пользователей память может закончиться
* При перезагрузке или сбое сервера - вся история исчезнет
* Чем длиннее история, тем больше токенов придется подавать на вход модели
* В случае платных моделей, это будет накладно по финансам
* Контекстное окно моделей тоже ограничено по количеству токенов </div>

<div class="alert alert-info" style="padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

Поэтому рекомендуется сохранять не всё подряд, а выделять основные сущности из диалога или периодически суммаризовать историю сообщений.</div>

<div style="padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

Для продакшен решений можно рассмотреть более практичные форматы хранения истории в `Langchain`, например, релизованы:
* `FileChatMessageHistory` - сохраняет историю взаимодействия сообщений в файл.
* `RedisChatMessageHistory` - сохраняет историю сообщений чата в базе данных Redis.
* `SQLChatMessageHistory` - сохраняет историю сообщений чата в базе данных SQL.
* `MongoDBChatMessageHistory` - в базе данных Mongo
* и многие [другие](https://python.langchain.com/api_reference/community/chat_message_histories.html)