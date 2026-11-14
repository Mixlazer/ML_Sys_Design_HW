# Данные Smart LLM Router

Скрипт `scripts/fetch_data.py` загружает в локальный кеш `raw/` полные языковые части:
MGSM (en/ru), Belebele (eng_Latn/rus_Cyrl), HumanEval (en).
Сырые файлы не публикуются в Git. Тексты при загрузке не меняются; TSV/JSONL
дополнительно сжимаются gzip.
`sources.lock.json` содержит закреплённые revisions, исходные URL, размеры и SHA-256
как upstream-байтов, так и хранимого контейнера. SHA-256 исходного содержимого проверяется перед EDA. Хеш контейнера служит
дополнительным ориентиром: gzip разных версий Python может отличаться заголовком.

| Данные | Авторство / источник | Лицензия |
| --- | --- | --- |
| MGSM | Shi et al., Google Research; Language Models are Multilingual Chain-of-Thought Reasoners | [CC BY 4.0](licenses/MGSM_CC_BY_4.0.txt) |
| Belebele | Bandarkar et al., Meta; The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants | [CC BY-SA 4.0](licenses/Belebele_CC_BY_SA_4.0.txt) |
| HumanEval | Chen et al., OpenAI; Evaluating Large Language Models Trained on Code | [MIT, copyright OpenAI](licenses/HumanEval_MIT.txt) |

Лицензии относятся к соответствующим данным, а не автоматически ко всему проекту.
Belebele используется для внешней диагностики, не обучения или настройки:
[указание upstream](https://github.com/facebookresearch/belebele#training-set).
Адаптированные материалы Belebele распространяются с сохранением CC BY-SA 4.0
и атрибуции. Код сборки обучающего набора Belebele с другой лицензией не копировался.

В локальном кеше HumanEval содержит canonical_solution и test как эталоны, однако они никогда
не входят в запрос LLM или признаки роутера. Скрипт EDA только разбирает AST.
Сырые новые пользовательские запросы, ответы провайдеров и ключи в Git не добавляются.

`reports/tables/task_metadata.csv` — производные признаки (ID, группа, источник,
домен, язык, размеры, флаги качества и иллюстративные split-метки), без условий и эталонов.
Колонка `tokens_proxy` измерена cl100k_base и не является usage выбранного endpoint.
`duplicate_options=True` означает карантин до проверки, а не автоматически неверный эталон.
Колонки split предназначены для исследования утечки; они не меняют роль внешнего benchmark.

Статьи: [MGSM](https://arxiv.org/abs/2210.03057),
[Belebele](https://aclanthology.org/2024.acl-long.44/),
[HumanEval](https://arxiv.org/abs/2107.03374).
