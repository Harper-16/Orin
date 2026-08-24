"""
ORIN v9
============================================================
COMPACT SYMBOLIC RAG + AUTOMATIC DATA MIGRATION

Major changes:
    - Removes Ollama completely
    - Automatically migrates old float-vector data.json
    - Converts knowledge to compact JSONL
    - Keeps symbolic encoding only for migration/training
    - Fixes float decoding
    - Atomic knowledge records
    - Deduplication
    - Aliases
    - Knowledge types
    - Better Wikipedia search
    - Groq extraction only
    - Incremental knowledge learning
    - Compact inverted RAG index
    - Raspberry Pi friendly

Directory:

data/
    data.json              <- old database, automatically migrated
    knowledge.jsonl        <- new database

data/train/
    data.txt
    words.jsonl
    rag_index.json
"""

import os
import sys
import ast
import json
import re
import math
import requests

from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")

OLD_RAG_DATA = os.path.join(
    DATA_DIR,
    "data.json"
)

KNOWLEDGE = os.path.join(
    DATA_DIR,
    "knowledge.jsonl"
)

TRAIN = os.path.join(
    TRAIN_DIR,
    "data.txt"
)

WORDS = os.path.join(
    TRAIN_DIR,
    "words.jsonl"
)

INDEX = os.path.join(
    TRAIN_DIR,
    "rag_index.json"
)


# ============================================================
# ONLINE SERVICES
# ============================================================

WIKI_API = "https://en.wikipedia.org/w/api.php"

GROQ_API = (
    "https://api.groq.com/openai/v1/chat/completions"
)

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# OLD SYMBOLIC ENCODING
#
# IMPORTANT:
# This exists ONLY so Orin can correctly migrate the old
# database.
# New knowledge is NOT stored this way.
# ============================================================

UPPER = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    0.8, 0.9, 0.10, 0.11, 0.12, 0.13,
    0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
    0.20, 0.21, 0.22, 0.23, 0.24, 0.25,
    0.26
]

LOWER = [
    0.11, 0.22, 0.33, 0.44, 0.55, 0.66,
    0.77, 0.88, 0.99, 0.1010, 0.1111,
    0.1212, 0.1313, 0.1414, 0.1515,
    0.1616, 0.1717, 0.1818, 0.1919,
    0.2020, 0.2121, 0.2222, 0.2323,
    0.2424, 0.2525, 0.2626
]

UPPER_MAP = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        UPPER
    )
)

LOWER_MAP = dict(
    zip(
        "abcdefghijklmnopqrstuvwxyz",
        LOWER
    )
)

ENCODE_MAP = {
    **UPPER_MAP,
    **LOWER_MAP
}

SPACE = 0.0
PUNCT = 0.01
UNKNOWN = 0.02


# ============================================================
# CRITICAL FLOAT DECODER FIX
# ============================================================

def old_decode_vector(vector):
    """
    Decode the OLD database format.

    We deliberately use explicit mappings instead of a
    collision-prone dictionary.
    """

    output = []

    for value in vector:

        try:
            x = float(value)
        except Exception:
            continue

        # Do NOT round first.
        # Compare using a tolerance.

        if abs(x - SPACE) < 1e-7:
            output.append(" ")
            continue

        if abs(x - PUNCT) < 1e-7:
            output.append(".")
            continue

        found = False

        # Lowercase first.
        for char, code in LOWER_MAP.items():

            if abs(
                x - float(code)
            ) < 1e-7:

                output.append(char)
                found = True
                break

        if found:
            continue

        # Uppercase.
        for char, code in UPPER_MAP.items():

            if abs(
                x - float(code)
            ) < 1e-7:

                output.append(char)
                found = True
                break

        if found:
            continue

        # UNKNOWN is deliberately ignored.
        if abs(x - UNKNOWN) < 1e-7:
            continue

    return "".join(output).strip()


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|"
    r"[.!?,;:!?()\[\]{}]"
)


def tokenize(text):
    return TOKEN_RE.findall(
        str(text)
    )


def clean(text):
    return str(text).lower().strip()


def normalize_text(text):
    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DATA MIGRATION
# ============================================================

class DataMigrator:

    FIELDS = (
        "topic",
        "subtopic1",
        "subtopic2",
        "subtopic3",
        "fact"
    )

    def decode_value(self, value):

        if isinstance(value, list):
            return old_decode_vector(value)

        return normalize_text(value)

    def old_record_to_new(self, record):

        if not isinstance(record, dict):
            return None

        decoded = {}

        for field in self.FIELDS:

            decoded[field] = self.decode_value(
                record.get(
                    field,
                    ""
                )
            )

        topic = decoded["topic"]
        fact = decoded["fact"]

        if not topic and not fact:
            return None

        if not fact:
            return None

        # ----------------------------------------------------
        # Remove the old UNKNOWN filler.
        # ----------------------------------------------------

        subtopics = []

        for field in (
            "subtopic1",
            "subtopic2",
            "subtopic3"
        ):

            value = normalize_text(
                decoded[field]
            )

            if (
                value
                and value not in subtopics
            ):
                subtopics.append(
                    value
                )

        # ----------------------------------------------------
        # Guess a knowledge type.
        # ----------------------------------------------------

        lower_fact = clean(
            fact
        )

        if any(
            x in lower_fact
            for x in (
                " is a ",
                " is an ",
                " is the ",
                " refers to ",
                " means ",
                " is defined as "
            )
        ):
            fact_type = "definition"

        elif any(
            x in lower_fact
            for x in (
                "because",
                "caused by",
                "due to",
                "results from"
            )
        ):
            fact_type = "cause"

        elif any(
            x in lower_fact
            for x in (
                "process",
                "works",
                "working",
                "method",
                "steps"
            )
        ):
            fact_type = "process"

        else:
            fact_type = "fact"

        return {
            "t": topic,
            "y": fact_type,
            "s": subtopics,
            "a": [],
            "f": normalize_text(fact)
        }

    def migrate(self):

        if not os.path.exists(
            OLD_RAG_DATA
        ):
            print(
                "[MIGRATE] No old data.json found."
            )
            return False

        if os.path.exists(
            KNOWLEDGE
        ):
            print(
                "[MIGRATE] knowledge.jsonl already exists."
            )
            print(
                "[MIGRATE] Skipping old database."
            )
            return False

        print()
        print("=" * 65)
        print("ORIN DATA MIGRATION")
        print("=" * 65)
        print()
        print(
            "[MIGRATE] Reading old data.json..."
        )

        try:

            with open(
                OLD_RAG_DATA,
                "r",
                encoding="utf-8"
            ) as f:

                old_data = json.load(f)

        except Exception as e:

            print(
                "[MIGRATE] Failed:",
                e
            )

            return False

        if not isinstance(
            old_data,
            list
        ):

            print(
                "[MIGRATE] data.json is not a list."
            )

            return False

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        seen = set()
        converted = 0
        duplicates = 0
        skipped = 0

        with open(
            KNOWLEDGE,
            "w",
            encoding="utf-8"
        ) as out:

            for old_record in old_data:

                record = self.old_record_to_new(
                    old_record
                )

                if not record:
                    skipped += 1
                    continue

                key = clean(
                    record["t"]
                    + "|"
                    + record["f"]
                )

                if key in seen:

                    duplicates += 1
                    continue

                seen.add(key)

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":"
                        )
                    )
                    + "\n"
                )

                converted += 1

        print(
            "[MIGRATE] Converted:",
            f"{converted:,}"
        )

        print(
            "[MIGRATE] Duplicates removed:",
            f"{duplicates:,}"
        )

        print(
            "[MIGRATE] Skipped:",
            f"{skipped:,}"
        )

        print(
            "[MIGRATE] New database:",
            KNOWLEDGE
        )

        print()
        print(
            "[MIGRATE] Old float encoding is now"
        )
        print(
            "[MIGRATE] OUT of the knowledge database."
        )

        print("=" * 65)
        print()

        return True


# ============================================================
# STOP WORDS
# ============================================================

STOP = {
    "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "of", "to", "in", "on",
    "at", "for", "from", "with", "by", "and", "or",
    "but", "this", "that", "these", "those", "it",
    "its", "as", "into", "than", "then", "there",
    "here", "about", "can", "could", "would", "should",
    "do", "does", "did", "will", "shall", "may",
    "might", "what", "who", "where", "when", "why",
    "how", "which", "i", "me", "my", "you", "your",
    "we", "our", "they", "their"
}


# ============================================================
# QUESTION INTENTS
# ============================================================

QUESTION_WORDS = {
    "what": "DEFINITION",
    "who": "PERSON",
    "where": "LOCATION",
    "when": "TIME",
    "why": "CAUSE",
    "how": "PROCESS",
    "which": "SELECTION"
}


# ============================================================
# ALIASES
# ============================================================

ALIASES = {
    "ai": "artificial intelligence",
    "a.i": "artificial intelligence",
    "machine intelligence": "artificial intelligence",

    "ml": "machine learning",

    "llm": "large language model",
    "gpt": "large language model",

    "nlp": "natural language processing",

    "sql": "structured query language",

    "pi": "raspberry pi",
    "raspberry pi": "raspberry pi",

    "math": "mathematics",
    "maths": "mathematics",

    "airplane": "aircraft",
    "aeroplane": "aircraft",

    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",

    "cars": "car",
    "vehicles": "vehicle"
}


def concepts(text):

    text = clean(text)

    found = []

    for alias, target in sorted(
        ALIASES.items(),
        key=lambda x: -len(x[0])
    ):

        if re.search(
            r"\b"
            + re.escape(alias)
            + r"\b",
            text
        ):

            if target not in found:
                found.append(target)

    return found


# ============================================================
# CHAT
# ============================================================

CHAT_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how r you",
    "who are you",
    "what are you",
    "thanks",
    "thank you",
    "thx",
    "bye",
    "goodbye"
}


def normalize_chat(text):

    return re.sub(
        r"[^a-z0-9 ]",
        "",
        clean(text)
    ).strip()


def is_chat(text):

    t = normalize_chat(
        text
    )

    if t in CHAT_PATTERNS:
        return True

    return (
        t.startswith("hi ")
        or
        t.startswith("hello ")
        or
        t.startswith("hey ")
    )


# ============================================================
# SENTENCE ANALYSIS
# ============================================================

def sentence_info(text):

    tokens = tokenize(
        text
    )

    words = [
        clean(x)
        for x in tokens
        if x not in ".!?,;:!?()[]{}"
    ]

    if not words:

        return {
            "type": "UNKNOWN",
            "intent": "UNKNOWN",
            "important": [],
            "question": None,
            "chat": False
        }

    if is_chat(text):

        return {
            "type": "CHAT",
            "intent": "CHAT",
            "important": [],
            "question": None,
            "chat": True
        }

    first = words[0]

    if (
        first in QUESTION_WORDS
        or
        "?" in tokens
    ):

        if first in QUESTION_WORDS:
            intent = QUESTION_WORDS[first]
        else:
            intent = "YES_NO"

        return {
            "type": "INTERROGATIVE",
            "intent": intent,
            "important": [
                w
                for w in words
                if w not in STOP
            ],
            "question": (
                first
                if first in QUESTION_WORDS
                else None
            ),
            "chat": False
        }

    imperative = {
        "open", "close", "find", "search",
        "show", "give", "tell", "explain",
        "describe", "define", "calculate",
        "list", "write", "create", "make",
        "get", "look", "compare", "use",
        "run", "start"
    }

    if first in imperative:

        return {
            "type": "IMPERATIVE",
            "intent": "ACTION",
            "important": [
                w
                for w in words
                if w not in STOP
            ],
            "question": None,
            "chat": False
        }

    return {
        "type": "DECLARATIVE",
        "intent": "STATEMENT",
        "important": [
            w
            for w in words
            if w not in STOP
        ],
        "question": None,
        "chat": False
    }


# ============================================================
# WORD MODEL
# ============================================================

class WordModel:

    def __init__(self):

        self.frequency = Counter()

        self.tags = defaultdict(
            Counter
        )

        self.words = set()

    def train(self, path):

        if not os.path.exists(path):

            print(
                "[WORDS] Training file missing:",
                path
            )

            return

        records = 0

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                # ------------------------------------------------
                # New plain-text format
                # ------------------------------------------------

                text = line

                # ------------------------------------------------
                # Old vector format
                # ------------------------------------------------

                if line.startswith("["):

                    try:

                        vector = ast.literal_eval(
                            line
                        )

                        if isinstance(
                            vector,
                            list
                        ):

                            text = old_decode_vector(
                                vector
                            )

                    except Exception:
                        continue

                tokens = tokenize(
                    text
                )

                for token in tokens:

                    if not token.isalnum():
                        continue

                    word = clean(
                        token
                    )

                    if not word:
                        continue

                    self.frequency[word] += 1
                    self.words.add(word)

                records += 1

        print(
            "[WORDS] Training records:",
            f"{records:,}"
        )

        print(
            "[WORDS] Vocabulary:",
            f"{len(self.words):,}"
        )

    def save(self, path):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            for word, frequency in (
                self.frequency.most_common()
            ):

                f.write(
                    json.dumps(
                        {
                            "w": word,
                            "f": frequency
                        },
                        separators=(
                            ",",
                            ":"
                        )
                    )
                    + "\n"
                )

        print(
            "[WORDS] Saved:",
            path
        )

    def load(self, path):

        if not os.path.exists(path):
            return False

        self.words = set()

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    # New JSON format.
                    if line.startswith("{"):

                        try:

                            d = json.loads(
                                line
                            )

                            word = d.get(
                                "w"
                            )

                            if word:
                                self.words.add(
                                    clean(word)
                                )

                            continue

                        except Exception:
                            pass

                    # Old format fallback.
                    if line.startswith("["):

                        try:

                            vector = ast.literal_eval(
                                line
                            )

                            word = old_decode_vector(
                                vector
                            )

                            if word:
                                self.words.add(
                                    clean(word)
                                )

                        except Exception:
                            pass

            return True

        except Exception as e:

            print(
                "[WORDS] Load error:",
                e
            )

            return False


# ============================================================
# RAG
# ============================================================

class RAG:

    def __init__(self):

        self.records = []

        self.index = defaultdict(
            list
        )

        self.phrases = defaultdict(
            list
        )

        self.df = Counter()

        self.total = 0

    # --------------------------------------------------------
    # LOAD KNOWLEDGE
    # --------------------------------------------------------

    def load_knowledge(self):

        self.records = []

        if not os.path.exists(
            KNOWLEDGE
        ):
            return False

        try:

            with open(
                KNOWLEDGE,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        d = json.loads(
                            line
                        )

                    except Exception:
                        continue

                    if (
                        isinstance(d, dict)
                        and d.get("f")
                    ):

                        self.records.append(
                            d
                        )

            self.total = len(
                self.records
            )

            return True

        except Exception as e:

            print(
                "[RAG] Knowledge load error:",
                e
            )

            return False

    # --------------------------------------------------------
    # BUILD INDEX
    # --------------------------------------------------------

    def build(self):

        if not self.load_knowledge():

            print(
                "[RAG] No knowledge database."
            )

            return False

        self.index = defaultdict(
            list
        )

        self.phrases = defaultdict(
            list
        )

        self.df = Counter()

        for idx, record in enumerate(
            self.records
        ):

            topic = clean(
                record.get(
                    "t",
                    ""
                )
            )

            subjects = " ".join(
                record.get(
                    "s",
                    []
                )
            )

            fact = clean(
                record.get(
                    "f",
                    ""
                )
            )

            aliases = " ".join(
                record.get(
                    "a",
                    []
                )
            )

            text = (
                topic
                + " "
                + subjects
                + " "
                + aliases
                + " "
                + fact
            )

            words = [
                w
                for w in tokenize(text)
                if w.isalnum()
                and w not in STOP
                and len(w) > 1
            ]

            unique = set(
                words
            )

            for word in unique:

                self.index[word].append(
                    idx
                )

                self.df[word] += 1

            for i in range(
                len(words) - 1
            ):

                phrase = (
                    words[i]
                    + " "
                    + words[i + 1]
                )

                self.phrases[
                    phrase
                ].append(idx)

        self.save_index()

        print(
            "[RAG] Records:",
            f"{self.total:,}"
        )

        print(
            "[RAG] Words:",
            f"{len(self.index):,}"
        )

        print(
            "[RAG] Phrases:",
            f"{len(self.phrases):,}"
        )

        return True

    # --------------------------------------------------------
    # SAVE INDEX
    # --------------------------------------------------------

    def save_index(self):

        os.makedirs(
            os.path.dirname(INDEX),
            exist_ok=True
        )

        data = {
            "index": dict(
                self.index
            ),
            "phrases": dict(
                self.phrases
            ),
            "df": dict(
                self.df
            ),
            "total": self.total
        }

        with open(
            INDEX,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                separators=(
                    ",",
                    ":"
                )
            )

    # --------------------------------------------------------
    # LOAD INDEX
    # --------------------------------------------------------

    def load(self):

        if not os.path.exists(
            KNOWLEDGE
        ):
            return False

        if not os.path.exists(
            INDEX
        ):
            return self.build()

        try:

            self.load_knowledge()

            with open(
                INDEX,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(
                    f
                )

            self.index = defaultdict(
                list,
                data.get(
                    "index",
                    {}
                )
            )

            self.phrases = defaultdict(
                list,
                data.get(
                    "phrases",
                    {}
                )
            )

            self.df = Counter(
                data.get(
                    "df",
                    {}
                )
            )

            self.total = data.get(
                "total",
                len(self.records)
            )

            return True

        except Exception:

            print(
                "[RAG] Index corrupted."
            )

            return self.build()

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        info,
        limit=6
    ):

        important = [
            clean(w)
            for w in info.get(
                "important",
                []
            )
            if clean(w)
            and clean(w) not in STOP
        ]

        concepts_found = info.get(
            "concepts",
            []
        )

        if (
            not important
            and not concepts_found
        ):
            return []

        candidates = set()

        # ----------------------------------------------------
        # Direct words
        # ----------------------------------------------------

        for word in important:

            for idx in self.index.get(
                word,
                []
            ):

                candidates.add(
                    idx
                )

        # ----------------------------------------------------
        # Phrases
        # ----------------------------------------------------

        for i in range(
            len(important) - 1
        ):

            phrase = (
                important[i]
                + " "
                + important[i + 1]
            )

            for idx in self.phrases.get(
                phrase,
                []
            ):

                candidates.add(
                    idx
                )

        # ----------------------------------------------------
        # Concepts
        # ----------------------------------------------------

        for concept in concepts_found:

            for word in concept.split():

                for idx in self.index.get(
                    word,
                    []
                ):

                    candidates.add(
                        idx
                    )

        scored = []

        N = max(
            1,
            self.total
        )

        query_set = set(
            important
        )

        for idx in candidates:

            if idx >= len(
                self.records
            ):
                continue

            record = self.records[idx]

            topic = clean(
                record.get(
                    "t",
                    ""
                )
            )

            subjects = clean(
                " ".join(
                    record.get(
                        "s",
                        []
                    )
                )
            )

            aliases = clean(
                " ".join(
                    record.get(
                        "a",
                        []
                    )
                )
            )

            fact = clean(
                record.get(
                    "f",
                    ""
                )
            )

            full = (
                topic
                + " "
                + subjects
                + " "
                + aliases
                + " "
                + fact
            )

            score = 0.0
            matched = 0

            # ------------------------------------------------
            # WORD MATCHING
            # ------------------------------------------------

            for word in important:

                if re.search(
                    r"\b"
                    + re.escape(word)
                    + r"\b",
                    full
                ):

                    matched += 1

                    df = self.df.get(
                        word,
                        N
                    )

                    idf = (
                        math.log(
                            (N + 1)
                            /
                            (df + 1)
                        )
                        + 1
                    )

                    score += (
                        8.0 * idf
                    )

                if re.search(
                    r"\b"
                    + re.escape(word)
                    + r"\b",
                    topic
                ):

                    score += 35.0

                if re.search(
                    r"\b"
                    + re.escape(word)
                    + r"\b",
                    subjects
                ):

                    score += 12.0

                if re.search(
                    r"\b"
                    + re.escape(word)
                    + r"\b",
                    fact
                ):

                    score += 6.0

            # ------------------------------------------------
            # PHRASE BOOST
            # ------------------------------------------------

            for i in range(
                len(important) - 1
            ):

                phrase = (
                    important[i]
                    + " "
                    + important[i + 1]
                )

                if phrase in full:
                    score += 25.0

                if phrase in topic:
                    score += 40.0

            # ------------------------------------------------
            # CONCEPT BOOST
            # ------------------------------------------------

            for concept in concepts_found:

                if concept in full:
                    score += 35.0

                if concept in topic:
                    score += 80.0

            # ------------------------------------------------
            # TYPE BOOST
            # ------------------------------------------------

            fact_type = record.get(
                "y",
                "fact"
            )

            intent = info.get(
                "intent",
                "UNKNOWN"
            )

            if (
                intent == "DEFINITION"
                and fact_type == "definition"
            ):
                score += 80.0

            elif (
                intent == "CAUSE"
                and fact_type == "cause"
            ):
                score += 50.0

            elif (
                intent == "PROCESS"
                and fact_type == "process"
            ):
                score += 50.0

            # ------------------------------------------------
            # DEFINITION QUALITY
            # ------------------------------------------------

            if intent == "DEFINITION":

                markers = (
                    " is a ",
                    " is an ",
                    " is the ",
                    " refers to ",
                    " means ",
                    " is defined as ",
                    " is the study of "
                )

                if any(
                    marker in fact
                    for marker in markers
                ):

                    score += 50.0

            # ------------------------------------------------
            # COVERAGE
            # ------------------------------------------------

            coverage = (
                matched
                /
                max(
                    1,
                    len(query_set)
                )
            )

            score += (
                coverage * 30.0
            )

            # ------------------------------------------------
            # RELEVANCE FILTER
            # ------------------------------------------------

            if matched == 0:
                continue

            scored.append(
                (
                    score,
                    idx
                )
            )

        scored.sort(
            reverse=True
        )

        return [
            self.records[idx]
            for score, idx
            in scored[:limit]
        ]


# ============================================================
# STITCHER
# ============================================================

class Stitcher:

    def clean_sentence(self, text):

        text = normalize_text(
            text
        )

        text = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            text
        )

        text = re.sub(
            r"([.!?]){2,}",
            r"\1",
            text
        )

        return text.strip()

    def stitch(
        self,
        records,
        info
    ):

        if not records:
            return ""

        selected = []
        seen = set()

        for record in records:

            fact = self.clean_sentence(
                record.get(
                    "f",
                    ""
                )
            )

            if not fact:
                continue

            key = re.sub(
                r"[^a-z0-9]",
                "",
                fact.lower()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            selected.append(
                fact
            )

            if (
                info.get(
                    "intent"
                )
                == "DEFINITION"
            ):

                if len(selected) >= 2:
                    break

            else:

                if len(selected) >= 3:
                    break

        return self.clean_sentence(
            " ".join(selected)
        )


# ============================================================
# CHECKER
# ============================================================

class Checker:

    def check(
        self,
        answer,
        records,
        info
    ):

        if not answer:
            return False, 0.0

        if not records:
            return False, 0.0

        answer_words = set(
            x
            for x in tokenize(
                answer.lower()
            )
            if x.isalnum()
        )

        query_words = set(
            x
            for x in info.get(
                "important",
                []
            )
            if x.isalnum()
            and x not in STOP
        )

        source_words = set()

        for record in records:

            text = (
                record.get("t", "")
                + " "
                + " ".join(
                    record.get(
                        "s",
                        []
                    )
                )
                + " "
                + record.get(
                    "f",
                    ""
                )
            )

            source_words.update(
                x
                for x in tokenize(
                    text.lower()
                )
                if x.isalnum()
            )

        query_coverage = (
            len(
                answer_words
                &
                query_words
            )
            /
            max(
                1,
                len(query_words)
            )
        )

        source_coverage = (
            len(
                answer_words
                &
                source_words
            )
            /
            max(
                1,
                len(answer_words)
            )
        )

        confidence = (
            query_coverage * 0.4
            +
            source_coverage * 0.6
        )

        confidence = min(
            1.0,
            confidence
        )

        grounded = (
            confidence >= 0.45
        )

        return (
            grounded,
            round(
                confidence,
                3
            )
        )


# ============================================================
# WIKIPEDIA
# ============================================================

class Wikipedia:

    def search(self, query):

        query = normalize_text(
            query
        )

        if not query:
            return None

        # ----------------------------------------------------
        # Remove question language.
        # ----------------------------------------------------

        words = [
            w
            for w in tokenize(
                query
            )
            if w.isalnum()
            and w.lower() not in {
                "what",
                "who",
                "where",
                "when",
                "why",
                "how",
                "which",
                "is",
                "are",
                "was",
                "were",
                "the",
                "a",
                "an",
                "about",
                "tell",
                "me",
                "explain",
                "define"
            }
        ]

        query = " ".join(
            words
        )

        if not query:
            return None

        print(
            "[Wikipedia] Query:",
            query
        )

        try:

            response = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": 8
                },
                timeout=15
            )

            if response.status_code != 200:

                print(
                    "[Wikipedia] HTTP:",
                    response.status_code
                )

                return None

            data = response.json()

            results = (
                data
                .get("query", {})
                .get("search", [])
            )

            if not results:
                return None

            # ------------------------------------------------
            # Prefer title containing the main query.
            # ------------------------------------------------

            query_words = set(
                words
            )

            best = None
            best_score = -1

            for result in results:

                title = normalize_text(
                    result.get(
                        "title",
                        ""
                    )
                )

                title_words = set(
                    tokenize(
                        title.lower()
                    )
                )

                score = len(
                    query_words
                    &
                    title_words
                )

                if clean(title) == clean(query):
                    score += 100

                if score > best_score:

                    best_score = score
                    best = title

            return best

        except Exception as e:

            print(
                "[Wikipedia] Search error:",
                e
            )

            return None

    def get(self, title):

        if not title:
            return ""

        try:

            response = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "titles": title,
                    "format": "json",
                    "utf8": 1,
                    "redirects": 1
                },
                timeout=20
            )

            if response.status_code != 200:
                return ""

            data = response.json()

            pages = (
                data
                .get("query", {})
                .get("pages", {})
            )

            for page in pages.values():

                text = page.get(
                    "extract",
                    ""
                )

                if text:
                    return text

            return ""

        except Exception as e:

            print(
                "[Wikipedia] Get error:",
                e
            )

            return ""


# ============================================================
# GROQ
# ============================================================

class GroqExtractor:

    def __init__(self):

        self.api_key = os.environ.get(
            "GROQ_API_KEY"
        )

    def extract(
        self,
        text,
        title
    ):

        if not self.api_key:

            print(
                "[Groq] GROQ_API_KEY is not set."
            )

            return []

        print(
            "[Groq] Extracting atomic knowledge..."
        )

        # ----------------------------------------------------
        # Keep the request manageable.
        # ----------------------------------------------------

        text = text[:16000]

        prompt = f"""
Extract useful factual knowledge about "{title}".

Return ONLY valid JSON:

{{
  "records": [
    {{
      "topic": "",
      "type": "",
      "subjects": [],
      "aliases": [],
      "fact": ""
    }}
  ]
}}

Rules:

1. Only use facts supported by the supplied text.
2. Never invent information.
3. Each fact must contain ONE main idea.
4. Keep facts concise.
5. topic should be the main concept.
6. type must be one of:
   definition
   fact
   cause
   effect
   process
   example
   history
   person
   location
   time
   formula
7. subjects should contain useful categories.
8. aliases should contain genuine alternate names only.
9. Do not duplicate facts.
10. Do not include markdown.
11. Do not explain your answer.

ARTICLE:
{text}
"""

        try:

            response = requests.post(
                GROQ_API,
                headers={
                    "Authorization":
                    "Bearer "
                    + self.api_key,

                    "Content-Type":
                    "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content":
                            "Return valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 3500,
                    "response_format": {
                        "type": "json_object"
                    }
                },
                timeout=120
            )

            print(
                "[Groq] HTTP:",
                response.status_code
            )

            if response.status_code != 200:

                print(
                    response.text[:500]
                )

                return []

            data = response.json()

            choices = data.get(
                "choices",
                []
            )

            if not choices:
                return []

            content = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )

            if not content:
                return []

            content = content.strip()

            content = re.sub(
                r"```json",
                "",
                content,
                flags=re.I
            )

            content = re.sub(
                r"```",
                "",
                content
            ).strip()

            try:

                parsed = json.loads(
                    content
                )

            except Exception:

                start = content.find(
                    "{"
                )

                end = content.rfind(
                    "}"
                )

                if (
                    start == -1
                    or
                    end == -1
                ):
                    return []

                parsed = json.loads(
                    content[
                        start:end + 1
                    ]
                )

            raw_records = parsed.get(
                "records",
                []
            )

            if not isinstance(
                raw_records,
                list
            ):
                return []

            records = []

            seen = set()

            for item in raw_records:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                topic = normalize_text(
                    item.get(
                        "topic",
                        title
                    )
                )

                fact = normalize_text(
                    item.get(
                        "fact",
                        ""
                    )
                )

                if not fact:
                    continue

                fact_type = clean(
                    item.get(
                        "type",
                        "fact"
                    )
                )

                allowed_types = {
                    "definition",
                    "fact",
                    "cause",
                    "effect",
                    "process",
                    "example",
                    "history",
                    "person",
                    "location",
                    "time",
                    "formula"
                }

                if fact_type not in allowed_types:
                    fact_type = "fact"

                subjects = item.get(
                    "subjects",
                    []
                )

                aliases = item.get(
                    "aliases",
                    []
                )

                if not isinstance(
                    subjects,
                    list
                ):
                    subjects = []

                if not isinstance(
                    aliases,
                    list
                ):
                    aliases = []

                subjects = [
                    normalize_text(x)
                    for x in subjects
                    if str(x).strip()
                ]

                aliases = [
                    normalize_text(x)
                    for x in aliases
                    if str(x).strip()
                ]

                key = clean(
                    topic
                    + "|"
                    + fact
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                records.append(
                    {
                        "t": topic,
                        "y": fact_type,
                        "s": subjects,
                        "a": aliases,
                        "f": fact
                    }
                )

            print(
                "[Groq] Extracted:",
                len(records),
                "atomic facts"
            )

            return records

        except requests.exceptions.Timeout:

            print(
                "[Groq] Request timed out."
            )

            return []

        except Exception as e:

            print(
                "[Groq] Error:",
                e
            )

            return []


# ============================================================
# LEARNER
# ============================================================

class Learner:

    def save_records(
        self,
        records
    ):

        if not records:
            return False

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        existing_keys = set()

        if os.path.exists(
            KNOWLEDGE
        ):

            with open(
                KNOWLEDGE,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    try:

                        d = json.loads(
                            line
                        )

                        key = clean(
                            d.get(
                                "t",
                                ""
                            )
                            + "|"
                            + d.get(
                                "f",
                                ""
                            )
                        )

                        existing_keys.add(
                            key
                        )

                    except Exception:
                        continue

        added = 0

        with open(
            KNOWLEDGE,
            "a",
            encoding="utf-8"
        ) as f:

            for record in records:

                key = clean(
                    record.get(
                        "t",
                        ""
                    )
                    + "|"
                    + record.get(
                        "f",
                        ""
                    )
                )

                if not key:
                    continue

                if key in existing_keys:
                    continue

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":"
                        )
                    )
                    + "\n"
                )

                existing_keys.add(
                    key
                )

                added += 1

        print(
            "[ORIN] Added:",
            added,
            "new records"
        )

        return added > 0


# ============================================================
# ORIN
# ============================================================

class Orin:

    def __init__(self):

        self.model = WordModel()

        self.rag = RAG()

        self.stitcher = Stitcher()

        self.checker = Checker()

        self.wikipedia = Wikipedia()

        self.groq = GroqExtractor()

        self.learner = Learner()

    # --------------------------------------------------------
    # FIRST START MIGRATION
    # --------------------------------------------------------

    def initialize(self):

        print()

        # -----------------------------------------------
        # Automatically migrate old data.json.
        # -----------------------------------------------

        migrator = DataMigrator()

        migrated = migrator.migrate()

        # -----------------------------------------------
        # Build/load RAG.
        # -----------------------------------------------

        if migrated:

            self.rag.build()

        elif not self.rag.load():

            print(
                "[ORIN] Building RAG..."
            )

            self.rag.build()

        print(
            "[ORIN] Initialization complete."
        )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    def analyze(
        self,
        prompt
    ):

        info = sentence_info(
            prompt
        )

        if info["chat"]:
            return (
                info,
                [],
                []
            )

        # -----------------------------------------------
        # Correct spelling conservatively.
        # -----------------------------------------------

        corrected = []

        corrections = []

        # Do not perform aggressive spelling correction
        # before we have a real vocabulary.
        vocabulary = self.model.words

        for token in tokenize(
            prompt
        ):

            if not token.isalnum():
                continue

            word = clean(
                token
            )

            corrected.append(
                word
            )

        search_text = " ".join(
            corrected
        )

        info["concepts"] = concepts(
            search_text
        )

        important = [
            word
            for word in corrected
            if word not in STOP
        ]

        # Add concept words.
        for concept in info["concepts"]:

            for word in concept.split():

                if word not in important:

                    important.append(
                        word
                    )

        info["important"] = important

        info["corrections"] = corrections

        records = self.rag.search(
            info,
            limit=6
        )

        return (
            info,
            records,
            corrections
        )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    def chat_response(
        self,
        prompt
    ):

        t = normalize_chat(
            prompt
        )

        if t in {
            "hi",
            "hello",
            "hey",
            "hiya"
        }:

            return (
                "Hi. What do you want to know?"
            )

        if t in {
            "how are you",
            "how r you"
        }:

            return (
                "I'm running normally. "
                "What do you want to know?"
            )

        if t in {
            "who are you",
            "what are you"
        }:

            return (
                "I'm Orin, a local symbolic "
                "knowledge system."
            )

        if t in {
            "thanks",
            "thank you",
            "thx"
        }:

            return "You're welcome."

        if t in {
            "bye",
            "goodbye"
        }:

            return "Goodbye."

        return None

    # --------------------------------------------------------
    # WIKIPEDIA LEARNING
    # --------------------------------------------------------

    def learn_from_wikipedia(
        self,
        prompt,
        info
    ):

        query = " ".join(
            info.get(
                "important",
                []
            )
        ).strip()

        if not query:
            return []

        print()
        print(
            "[ORIN] Local knowledge insufficient."
        )

        print(
            "[ORIN] Searching Wikipedia..."
        )

        title = self.wikipedia.search(
            query
        )

        if not title:

            print(
                "[ORIN] Wikipedia found nothing."
            )

            return []

        print(
            "[ORIN] Wikipedia:",
            title
        )

        article = self.wikipedia.get(
            title
        )

        if not article:

            print(
                "[ORIN] Could not retrieve article."
            )

            return []

        records = self.groq.extract(
            article,
            title
        )

        if not records:
            return []

        if self.learner.save_records(
            records
        ):

            print(
                "[ORIN] Knowledge saved."
            )

            print(
                "[ORIN] Updating RAG..."
            )

            self.rag.build()

        return records

    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

    def ask(
        self,
        prompt
    ):

        prompt = prompt.strip()

        if not prompt:
            return

        chat = self.chat_response(
            prompt
        )

        if chat:

            print()
            print(
                "[ORIN]",
                chat
            )
            print()

            return

        info, records, corrections = (
            self.analyze(
                prompt
            )
        )

        answer = self.stitcher.stitch(
            records,
            info
        )

        grounded = False
        confidence = 0.0

        if answer:

            grounded, confidence = (
                self.checker.check(
                    answer,
                    records,
                    info
                )
            )

        # -----------------------------------------------
        # Decide whether local result is good enough.
        # -----------------------------------------------

        local_good = (
            bool(answer)
            and grounded
            and confidence >= 0.50
        )

        source = "LOCAL"

        # -----------------------------------------------
        # Wikipedia fallback.
        # -----------------------------------------------

        if not local_good:

            learned = (
                self.learn_from_wikipedia(
                    prompt,
                    info
                )
            )

            if learned:

                records = self.rag.search(
                    info,
                    limit=6
                )

                answer = self.stitcher.stitch(
                    records,
                    info
                )

                if answer:

                    grounded, confidence = (
                        self.checker.check(
                            answer,
                            records,
                            info
                        )
                    )

                    source = "WIKIPEDIA + GROQ"

        if not answer:

            source = "NONE"

            grounded = False

            confidence = 0.0

            answer = (
                "I could not find enough "
                "reliable information."
            )

        # -----------------------------------------------
        # OUTPUT
        # -----------------------------------------------

        print()
        print("=" * 65)
        print("ORIN v9")
        print("=" * 65)
        print()

        print(
            "Sentence Type :",
            info["type"]
        )

        print(
            "Intent        :",
            info["intent"]
        )

        print(
            "Concepts      :",
            ", ".join(
                info.get(
                    "concepts",
                    []
                )
            )
            or
            "none"
        )

        print(
            "Important     :",
            ", ".join(
                info.get(
                    "important",
                    []
                )
            )
            or
            "none"
        )

        print(
            "Corrections   :",
            ", ".join(
                corrections
            )
            or
            "none"
        )

        print(
            "Retrieved     :",
            len(records)
        )

        print(
            "Source        :",
            source
        )

        print(
            "Grounded      :",
            grounded
        )

        print(
            "Confidence    :",
            confidence
        )

        print("-" * 65)

        print(
            answer
        )

        print("=" * 65)
        print()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def train(self):

        print()
        print("=" * 65)
        print("ORIN v9 COMPILATION")
        print("=" * 65)
        print()

        # -----------------------------------------------
        # Migrate first.
        # -----------------------------------------------

        DataMigrator().migrate()

        # -----------------------------------------------
        # Train symbolic vocabulary.
        # -----------------------------------------------

        print(
            "[1/2] Compiling symbolic vocabulary..."
        )

        self.model.train(
            TRAIN
        )

        self.model.save(
            WORDS
        )

        # -----------------------------------------------
        # RAG.
        # -----------------------------------------------

        print()
        print(
            "[2/2] Compiling compact RAG..."
        )

        self.rag.build()

        print()
        print("=" * 65)
        print("ORIN v9 COMPILATION COMPLETE")
        print("=" * 65)
        print()

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    def chat(self):

        self.initialize()

        # -----------------------------------------------
        # Load vocabulary if present.
        # -----------------------------------------------

        self.model.load(
            WORDS
        )

        print()
        print("=" * 65)
        print("ORIN v9 CHAT")
        print("=" * 65)
        print()

        print(
            "Compact local RAG + Wikipedia learning"
        )

        print(
            "Ollama: disabled / removed"
        )

        print(
            "Type 'exit' to quit."
        )

        print()

        while True:

            try:

                prompt = input(
                    "You: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError
            ):

                print()
                print(
                    "[ORIN] Goodbye."
                )

                break

            if not prompt:
                continue

            if normalize_chat(
                prompt
            ) in {
                "exit",
                "quit"
            }:

                print(
                    "[ORIN] Goodbye."
                )

                break

            self.ask(
                prompt
            )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    engine = Orin()

    if len(sys.argv) >= 2:

        if sys.argv[1] == "--train":

            engine.train()
            return

        if (
            sys.argv[1] == "--ask"
            and
            len(sys.argv) >= 3
        ):

            engine.initialize()

            engine.ask(
                " ".join(
                    sys.argv[2:]
                )
            )

            return

        if sys.argv[1] == "--chat":

            engine.chat()
            return

    engine.chat()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()