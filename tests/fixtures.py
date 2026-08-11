"""Test fixtures: realistic sentences by language for unit tests."""

# English: clear sentences with proper punctuation
EN_SINGLE_SENTENCE = "Artificial intelligence has revolutionized many industries."

EN_MULTIPLE_SENTENCES = (
    "Machine learning is a subset of artificial intelligence. "
    "It focuses on learning patterns from data. "
    "Neural networks are inspired by the human brain."
)

EN_PARAGRAPH = (
    "The development of artificial intelligence has accelerated significantly. "
    "Researchers are now working on creating more robust and interpretable models. "
    "Challenges remain in areas such as fairness, privacy, and explainability. "
    "The field continues to evolve rapidly."
)

# Spanish: multiple sentences
ES_SINGLE_SENTENCE = "La inteligencia artificial ha revolucionado muchas industrias."

ES_MULTIPLE_SENTENCES = (
    "El aprendizaje automático es un subconjunto de inteligencia artificial. "
    "Se enfoca en aprender patrones de los datos. "
    "Las redes neuronales están inspiradas en el cerebro humano."
)

ES_PARAGRAPH = (
    "El desarrollo de la inteligencia artificial se ha acelerado significativamente. "
    "Los investigadores ahora trabajan en crear modelos más robustos e interpretables. "
    "Los desafíos persisten en áreas como equidad, privacidad y explicabilidad. "
    "El campo continúa evolucionando rápidamente."
)

# Portuguese: multiple sentences
PT_SINGLE_SENTENCE = "A inteligência artificial revolucionou muitas indústrias."

PT_MULTIPLE_SENTENCES = (
    "O aprendizado de máquina é um subconjunto da inteligência artificial. "
    "Ele se concentra em aprender padrões dos dados. "
    "As redes neurais são inspiradas pelo cérebro humano."
)

PT_PARAGRAPH = (
    "O desenvolvimento da inteligência artificial acelerou significativamente. "
    "Os pesquisadores agora trabalham na criação de modelos mais robustos e interpretáveis. "
    "Os desafios persistem em áreas como equidade, privacidade e explicabilidade. "
    "O campo continua evoluindo rapidamente."
)

# French: multiple sentences
FR_SINGLE_SENTENCE = "L'intelligence artificielle a révolutionné de nombreuses industries."

FR_MULTIPLE_SENTENCES = (
    "L'apprentissage automatique est un sous-ensemble de l'intelligence artificielle. "
    "Il se concentre sur l'apprentissage de modèles à partir des données. "
    "Les réseaux de neurones s'inspirent du cerveau humain."
)

FR_PARAGRAPH = (
    "Le développement de l'intelligence artificielle s'est considérablement accéléré. "
    "Les chercheurs travaillent maintenant à la création de modèles plus robustes et interprétables. "
    "Des défis subsistent dans les domaines de l'équité, la confidentialité et l'explicabilité. "
    "Le domaine continue d'évoluer rapidement."
)

# Chinese: using Chinese sentence-ending punctuation
ZH_SINGLE_SENTENCE = "人工智能已经彻底改变了许多行业。"

ZH_MULTIPLE_SENTENCES = "机器学习是人工智能的一个子集。它专注于从数据中学习模式。神经网络受到人脑的启发。"

ZH_PARAGRAPH = (
    "人工智能的发展已经大大加速。"
    "研究人员现在正在努力创建更强大和可解释的模型。"
    "公平性、隐私性和可解释性等领域仍然存在挑战。"
    "该领域继续快速发展。"
)

# Arabic: using Arabic sentence-ending punctuation
AR_SINGLE_SENTENCE = "غيرت الذكاء الاصطناعي العديد من الصناعات بشكل كبير."

AR_MULTIPLE_SENTENCES = (
    "التعلم الآلي هو مجموعة فرعية من الذكاء الاصطناعي. "
    "يركز على تعلم الأنماط من البيانات. "
    "الشبكات العصبية مستوحاة من الدماغ البشري."
)

AR_PARAGRAPH = (
    "تسارع تطور الذكاء الاصطناعي بشكل كبير. "
    "يعمل الباحثون الآن على إنشاء نماذج أكثر قوة وقابلية للتفسير. "
    "تبقى التحديات قائمة في مجالات مثل الإنصاف والخصوصية والقابلية للتفسير. "
    "يستمر المجال في التطور السريع."
)

# Russian: multiple sentences
RU_SINGLE_SENTENCE = "Искусственный интеллект произвел революцию во многих отраслях."

RU_MULTIPLE_SENTENCES = (
    "Машинное обучение является подмножеством искусственного интеллекта. "
    "Оно сосредоточено на изучении закономерностей в данных. "
    "Нейронные сети вдохновлены человеческим мозгом."
)

RU_PARAGRAPH = (
    "Развитие искусственного интеллекта значительно ускорилось. "
    "Исследователи теперь работают над созданием более надежных и интерпретируемых моделей. "
    "Проблемы остаются в таких областях, как справедливость, конфиденциальность и объяснимость. "
    "Область продолжает развиваться быстрыми темпами."
)

# Korean: using ASCII punctuation (Korean text with mixed punctuation)
KO_SINGLE_SENTENCE = "인공지능은 많은 산업에 혁명을 일으켰습니다."

KO_MULTIPLE_SENTENCES = (
    "머신러닝은 인공지능의 부분집합입니다. "
    "데이터에서 패턴을 학습하는 데 중점을 두고 있습니다. "
    "신경망은 인간 뇌에서 영감을 받습니다."
)

KO_PARAGRAPH = (
    "인공지능의 발전은 크게 가속화되었습니다. "
    "연구자들은 이제 더욱 강력하고 해석 가능한 모델을 만드는 데 노력하고 있습니다. "
    "공정성, 개인정보 보호, 설명 가능성 등의 영역에서 여전히 과제가 남아 있습니다. "
    "이 분야는 계속 빠르게 진화하고 있습니다."
)

# Quechua: minimal corpus, will use ASCII fallback
QU_SINGLE_SENTENCE = "Allinchay rimayniyki."

QU_MULTIPLE_SENTENCES = "Allinchay rimayniyki. Hukpacha rimayniyki. Queshapi rimayniyki."

# Malayan: mixed script, ASCII fallback
MS_SINGLE_SENTENCE = "Kecerdasan buatan telah merevolusi banyak industri."

MS_MULTIPLE_SENTENCES = (
    "Pembelajaran mesin adalah bagian dari kecerdasan buatan. "
    "Ini berfokus pada pembelajaran pola dari data. "
    "Jaringan saraf terinspirasi oleh otak manusia."
)

# Edge cases for robustness testing
EMPTY_STRING = ""
WHITESPACE_ONLY = "   \n  \t  "
SINGLE_WORD = "Hello"
SENTENCE_WITHOUT_PERIOD = "This is a sentence without ending punctuation"
VERY_LONG_SENTENCE = (
    "This is an extremely long sentence that continues and continues and continues "
    "without any punctuation for a very long time which would exceed typical chunk "
    "sizes if we were to use this as test data for the chunking algorithm. "
    "It contains multiple clauses and thoughts all run together."
)
SENTENCE_WITH_NUMBERS = "The experiment showed 87.5% accuracy with p < 0.001."
SENTENCE_WITH_ELLIPSIS = "The findings suggest that there might be more... interesting patterns ahead."
SENTENCE_WITH_QUOTES = 'The speaker said, "Machine learning is transforming society."'
SENTENCE_WITH_URLS = "See more at https://example.com/ai. Also check https://research.org."

# Fixture dictionary for parametrized tests
LANGUAGE_FIXTURES = {
    "en": {
        "single": EN_SINGLE_SENTENCE,
        "multiple": EN_MULTIPLE_SENTENCES,
        "paragraph": EN_PARAGRAPH,
    },
    "es": {
        "single": ES_SINGLE_SENTENCE,
        "multiple": ES_MULTIPLE_SENTENCES,
        "paragraph": ES_PARAGRAPH,
    },
    "pt": {
        "single": PT_SINGLE_SENTENCE,
        "multiple": PT_MULTIPLE_SENTENCES,
        "paragraph": PT_PARAGRAPH,
    },
    "fr": {
        "single": FR_SINGLE_SENTENCE,
        "multiple": FR_MULTIPLE_SENTENCES,
        "paragraph": FR_PARAGRAPH,
    },
    "zh": {
        "single": ZH_SINGLE_SENTENCE,
        "multiple": ZH_MULTIPLE_SENTENCES,
        "paragraph": ZH_PARAGRAPH,
    },
    "ar": {
        "single": AR_SINGLE_SENTENCE,
        "multiple": AR_MULTIPLE_SENTENCES,
        "paragraph": AR_PARAGRAPH,
    },
    "ru": {
        "single": RU_SINGLE_SENTENCE,
        "multiple": RU_MULTIPLE_SENTENCES,
        "paragraph": RU_PARAGRAPH,
    },
    "ko": {
        "single": KO_SINGLE_SENTENCE,
        "multiple": KO_MULTIPLE_SENTENCES,
        "paragraph": KO_PARAGRAPH,
    },
    "ms": {
        "single": MS_SINGLE_SENTENCE,
        "multiple": MS_MULTIPLE_SENTENCES,
    },
    "qu": {
        "single": QU_SINGLE_SENTENCE,
        "multiple": QU_MULTIPLE_SENTENCES,
    },
}
