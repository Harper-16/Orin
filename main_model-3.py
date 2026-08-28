import os
import sys
import json
import math
import re
import struct
import requests
from collections import Counter,defaultdict
BASE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE,"data")
TRAIN_DIR=os.path.join(DATA_DIR,"train")
KNOWLEDGE=os.path.join(DATA_DIR,"knowledge.jsonl")
TRAIN=os.path.join(TRAIN_DIR,"data.txt")
WORDS=os.path.join(TRAIN_DIR,"words.jsonl")
RAG_FILE=os.path.join(TRAIN_DIR,"rag_index.json")
SEMANTIC=os.path.join(TRAIN_DIR,"semantic_index.f32")
CACHE_FILE=os.path.join(TRAIN_DIR,"answer_cache.json")
WIKI_API="https://en.wikipedia.org/w/api.php"
GROQ_API="https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL="openai/gpt-oss-20b"
DIM=384
TOKEN_RE=re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[.!?,;:!?()\[\]{}]")
STOP={"the","a","an","is","are","was","were","be","been","being","of","to","in","on","at","for","from","with","by","and","or","but","this","that","these","those","it","its","as","into","than","then","there","here","about","can","could","would","should","do","does","did","will","shall","may","might","what","who","where","when","why","how","which","i","me","my","you","your","we","our","they","their","tell","please"}
QUESTION_WORDS={"what":"DEFINITION","who":"PERSON","where":"LOCATION","when":"TIME","why":"CAUSE","how":"PROCESS","which":"SELECTION"}
ACTION_WORDS={"open","close","find","search","show","give","tell","explain","describe","define","calculate","list","write","create","make","get","look","compare","use","run","start"}
ALIASES={"ai":"artificial intelligence","a.i":"artificial intelligence","machine intelligence":"artificial intelligence","ml":"machine learning","llm":"large language model","gpt":"large language model","nlp":"natural language processing","sql":"structured query language","pi":"raspberry pi","math":"mathematics","maths":"mathematics","mathematic":"mathematics","airplane":"aircraft","aeroplane":"aircraft","trucks":"truck","lorry":"truck","lorries":"truck","cars":"car","vehicles":"vehicle","dhoni":"mahendra singh dhoni","m s dhoni":"mahendra singh dhoni","ms dhoni":"mahendra singh dhoni","sachin":"sachin tendulkar","einstein":"albert einstein","spiderman":"spider-man","spider man":"spider-man"}
CHAT={"hi","hello","hey","hiya","yo","sup","good morning","good afternoon","good evening","how are you","how r you","who are you","what are you","thanks","thank you","thx","bye","goodbye","whats up","what's up","wassup"}
def tokenize(x):
    return TOKEN_RE.findall(str(x))
def clean(x):
    return str(x).lower().strip()
def normalize(x):
    return re.sub(r"\s+"," ",str(x)).strip()
def alnum_words(x):
    return [clean(w) for w in tokenize(x) if w.isalnum()]
def compact(x):
    return re.sub(r"[^a-z0-9]","",clean(x))
def chat_clean(x):
    return re.sub(r"[^a-z0-9 ]","",clean(x)).strip()
def is_chat(x):
    t=chat_clean(x)
    return t in CHAT or t.startswith("hi ") or t.startswith("hello ") or t.startswith("hey ")
def concepts(text):
    text=clean(text)
    found=[]
    for a,t in sorted(ALIASES.items(),key=lambda x:-len(x[0])):
        if re.search(r"\b"+re.escape(a)+r"\b",text) and t not in found:
            found.append(t)
    return found
def analyze_sentence(text):
    tokens=tokenize(text)
    words=[clean(x) for x in tokens if x not in ".!?,;:!?()[]{}"]
    if not words:
        return {"type":"UNKNOWN","intent":"UNKNOWN","important":[],"question":None,"chat":False}
    if is_chat(text):
        return {"type":"CHAT","intent":"CHAT","important":[],"question":None,"chat":True}
    first=words[0]
    if first in QUESTION_WORDS or "?" in tokens:
        intent=QUESTION_WORDS.get(first,"YES_NO")
        return {"type":"INTERROGATIVE","intent":intent,"important":[w for w in words if w not in STOP],"question":first if first in QUESTION_WORDS else None,"chat":False}
    if first in ACTION_WORDS:
        return {"type":"IMPERATIVE","intent":"ACTION","important":[w for w in words if w not in STOP],"question":None,"chat":False}
    return {"type":"DECLARATIVE","intent":"STATEMENT","important":[w for w in words if w not in STOP],"question":None,"chat":False}
class WordModel:
    def __init__(self):
        self.frequency=Counter()
        self.words=set()
    def train(self,path):
        if not os.path.exists(path):
            print("[WORDS] Training file missing:",path)
            return
        records=0
        with open(path,encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                text=line
                if line.startswith("["):
                    try:
                        v=json.loads(line.replace("'","\""))
                        if isinstance(v,list):
                            text=self.old_decode(v)
                    except:
                        continue
                for token in tokenize(text):
                    if token.isalnum():
                        w=clean(token)
                        self.frequency[w]+=1
                        self.words.add(w)
                records+=1
        print("[WORDS] Training records:",f"{records:,}")
        print("[WORDS] Vocabulary:",f"{len(self.words):,}")
    def old_decode(self,v):
        out=[]
        upper=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26]
        lower=[0.11,0.22,0.33,0.44,0.55,0.66,0.77,0.88,0.99,0.1010,0.1111,0.1212,0.1313,0.1414,0.1515,0.1616,0.1717,0.1818,0.1919,0.2020,0.2121,0.2222,0.2323,0.2424,0.2525,0.2626]
        um=dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",upper))
        lm=dict(zip("abcdefghijklmnopqrstuvwxyz",lower))
        for x in v:
            try:x=float(x)
            except:continue
            if abs(x)<1e-7:
                out.append(" ")
                continue
            if abs(x-.01)<1e-7:
                out.append(".")
                continue
            found=False
            for c,n in lm.items():
                if abs(x-float(n))<1e-7:
                    out.append(c)
                    found=True
                    break
            if found:
                continue
            for c,n in um.items():
                if abs(x-float(n))<1e-7:
                    out.append(c)
                    break
        return "".join(out).strip()
    def save(self,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"w",encoding="utf-8") as f:
            for w,n in self.frequency.most_common():
                f.write(json.dumps({"w":w,"f":n},separators=(",",":"))+"\n")
    def load(self,path):
        if not os.path.exists(path):
            return False
        try:
            with open(path,encoding="utf-8") as f:
                for line in f:
                    try:
                        d=json.loads(line)
                        if d.get("w"):
                            self.words.add(clean(d["w"]))
                    except:
                        pass
            return True
        except:
            return False
class FastSemantic384:
    def __init__(self):
        self.dim=DIM
        self.cache={}
    def word_vector(self,word):
        word=clean(word)
        cached=self.cache.get(word)
        if cached is not None:
            return cached
        seed=0
        for i,c in enumerate(word):
            seed=(seed+ord(c)*(i+1))&0xffffffff
        v=[]
        for i in range(self.dim):
            x=math.sin(seed+i*1.61803398875)*math.cos(seed-i*.70710678118)
            v.append(x)
        self.cache[word]=v
        return v
    def vectorize(self,text):
        words=[clean(w) for w in tokenize(text) if w.isalnum() and clean(w) not in STOP]
        if not words:
            return [0.0]*self.dim
        v=[0.0]*self.dim
        for word in words:
            wv=self.word_vector(word)
            for i in range(self.dim):
                v[i]+=wv[i]
        inv=1.0/len(words)
        mag=0.0
        for i in range(self.dim):
            v[i]*=inv
            mag+=v[i]*v[i]
        mag=math.sqrt(mag)
        if mag:
            inv=1.0/mag
            for i in range(self.dim):
                v[i]*=inv
        return v
    def similarity(self,a,b):
        return sum(x*y for x,y in zip(a,b))
class SemanticStore:
    def __init__(self):
        self.data=None
        self.count=0
    def load(self,path):
        if not os.path.exists(path):
            return False
        try:
            size=os.path.getsize(path)
            if size<DIM*4:
                return False
            with open(path,"rb") as f:
                self.data=f.read()
            self.count=len(self.data)//(DIM*4)
            return True
        except:
            return False
    def vector(self,index):
        start=index*DIM*4
        end=start+DIM*4
        if not self.data or end>len(self.data):
            return None
        return struct.unpack_from("<384f",self.data,start)
    def save(self,path,vectors):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"wb") as f:
            for v in vectors:
                f.write(struct.pack("<384f",*v))
class RAG:
    def __init__(self):
        self.records=[]
        self.index=defaultdict(list)
        self.df=Counter()
        self.total=0
    def load_knowledge(self):
        self.records=[]
        if not os.path.exists(KNOWLEDGE):
            return False
        try:
            with open(KNOWLEDGE,encoding="utf-8") as f:
                for line in f:
                    try:
                        d=json.loads(line)
                        if isinstance(d,dict) and d.get("f"):
                            self.records.append(d)
                    except:
                        pass
            self.total=len(self.records)
            return bool(self.records)
        except:
            return False
    def record_text(self,r):
        return normalize(" ".join([str(r.get("t",""))," ".join(r.get("s",[]))," ".join(r.get("a",[])),str(r.get("f",""))])).lower()
    def build(self):
        if not self.load_knowledge():
            print("[RAG] No knowledge database.")
            return False
        self.index=defaultdict(list)
        self.df=Counter()
        for idx,r in enumerate(self.records):
            words=[w for w in tokenize(self.record_text(r)) if w.isalnum() and w not in STOP and len(w)>1]
            for word in set(words):
                self.index[word].append(idx)
                self.df[word]+=1
        os.makedirs(os.path.dirname(RAG_FILE),exist_ok=True)
        with open(RAG_FILE,"w",encoding="utf-8") as f:
            json.dump({"index":dict(self.index),"df":dict(self.df),"total":len(self.records)},f,separators=(",",":"))
        print("[ORIN] Records:",len(self.records))
        print("[ORIN] Index words:",len(self.index))
        return True
    def load(self):
        if not self.load_knowledge():
            return False
        if not os.path.exists(RAG_FILE):
            return self.build()
        try:
            with open(RAG_FILE,encoding="utf-8") as f:
                data=json.load(f)
            self.index=defaultdict(list,data.get("index",{}))
            self.df=Counter(data.get("df",{}))
            return True
        except:
            return self.build()
    def candidates(self,info):
        c=set()
        for w in info.get("important",[]):
            c.update(self.index.get(w,[]))
        for phrase in info.get("search_terms",[]):
            for w in alnum_words(phrase):
                c.update(self.index.get(w,[]))
        return c
    def exact_topic(self,info):
        target=clean(info.get("target",""))
        if not target:
            return []
        tc=compact(target)
        exact=[]
        partial=[]
        for i,r in enumerate(self.records):
            topic=clean(r.get("t",""))
            aliases=[clean(x) for x in r.get("a",[])]
            if compact(topic)==tc or any(compact(x)==tc for x in aliases):
                exact.append(i)
            elif tc and (tc in compact(topic) or compact(topic) in tc):
                partial.append(i)
        return exact or partial
    def search(self,info,semantic,store,limit=10):
        exact=self.exact_topic(info)
        if exact:
            ranked=self.rank_indexes(exact,info,semantic,store)
            return [self.records[i] for i in ranked[:limit]]
        candidates=self.candidates(info)
        if not candidates:
            return []
        ranked=self.rank_indexes(list(candidates),info,semantic,store)
        return [self.records[i] for i in ranked[:limit]]
    def rank_indexes(self,indices,info,semantic,store):
        qwords=set(info.get("important",[]))
        target=clean(info.get("target",""))
        target_compact=compact(target)
        intent=info.get("intent")
        N=max(1,self.total)
        scores=[]
        for idx in indices:
            if idx>=len(self.records):
                continue
            r=self.records[idx]
            topic=clean(r.get("t",""))
            aliases=clean(" ".join(r.get("a",[])))
            subjects=clean(" ".join(r.get("s",[])))
            fact=clean(r.get("f",""))
            full=topic+" "+aliases+" "+subjects+" "+fact
            score=0.0
            matched=0
            topic_compact=compact(topic)
            if target_compact:
                if topic_compact==target_compact:
                    score+=1000
                elif any(compact(x)==target_compact for x in r.get("a",[])):
                    score+=900
                elif target_compact in topic_compact or topic_compact in target_compact:
                    score+=500
                elif target_compact in compact(full):
                    score+=120
            for w in qwords:
                if re.search(r"\b"+re.escape(w)+r"\b",full):
                    matched+=1
                    df=self.df.get(w,N)
                    score+=6*(math.log((N+1)/(df+1))+1)
                if re.search(r"\b"+re.escape(w)+r"\b",topic):
                    score+=70
                if re.search(r"\b"+re.escape(w)+r"\b",aliases):
                    score+=50
                if re.search(r"\b"+re.escape(w)+r"\b",subjects):
                    score+=20
            if intent=="DEFINITION":
                typ=clean(r.get("y",""))
                if typ=="definition":
                    score+=300
                if re.search(r"\b(is a|is an|is the|refers to|means|defined as|is known as)\b",fact):
                    score+=180
                if re.search(r"\b(is|are)\b",fact[:120]):
                    score+=35
                if typ in {"example","effect","history","time","location"}:
                    score-=90
                if len(fact)<25:
                    score-=80
            elif intent=="PERSON":
                if clean(r.get("y",""))=="person":
                    score+=300
                if re.search(r"\b(born|died|cricketer|actor|scientist|player|politician|engineer|president|author|founder|physicist|mathematician)\b",fact):
                    score+=100
                if topic_compact==target_compact:
                    score+=500
            elif intent=="CAUSE" and clean(r.get("y",""))=="cause":
                score+=250
            elif intent=="PROCESS" and clean(r.get("y",""))=="process":
                score+=250
            if matched:
                score+=45*(matched/max(1,len(qwords)))
            scores.append([score,idx])
        scores.sort(key=lambda x:x[0],reverse=True)
        # Only use the expensive semantic layer on the strongest symbolic candidates.
        top=scores[:30]
        if len(top)>1:
            qtext=target or " ".join(info.get("important",[]))
            qvec=semantic.vectorize(qtext)
            reranked=[]
            for score,idx in top:
                if idx<store.count:
                    sv=store.vector(idx)
                    if sv:
                        sim=semantic.similarity(qvec,sv)
                        # Semantic score is deliberately small so it cannot defeat exact-topic matching.
                        score+=sim*35
                reranked.append([score,idx])
            reranked.sort(key=lambda x:x[0],reverse=True)
            scores=reranked+scores[30:]
        return [idx for _,idx in scores]
class Stitcher:
    def clean(self,x):
        x=normalize(x)
        x=re.sub(r"\s+([,.!?;:])",r"\1",x)
        x=re.sub(r"([.!?]){2,}",r"\1",x)
        x=re.sub(r"\.{2,}",".",x)
        x=re.sub(r"\s+\.",".",x)
        x=re.sub(r"\s+,\s*",", ",x)
        return x.strip()
    def quality(self,fact,info):
        f=clean(fact)
        if len(f)<25:
            return -100
        score=0
        target=compact(info.get("target",""))
        fc=compact(f)
        if target and target in fc:
            score+=50
        if info["intent"]=="DEFINITION":
            if re.search(r"\b(is a|is an|is the|refers to|means|defined as)\b",f):
                score+=100
            if re.search(r"\b(example|customers|users want|historical game studies)\b",f):
                score-=70
        return score
    def choose(self,records,info):
        if not records:
            return ""
        ranked=[]
        for i,r in enumerate(records):
            f=self.clean(r.get("f",""))
            ranked.append((self.quality(f,info)-i*2,f))
        ranked.sort(reverse=True)
        if info["intent"] in {"DEFINITION","PERSON","LOCATION","TIME"}:
            return ranked[0][1] if ranked else ""
        selected=[]
        seen=set()
        for _,f in ranked:
            key=compact(f)
            if f and key not in seen:
                selected.append(f)
                seen.add(key)
            if len(selected)>=2:
                break
        return self.clean(" ".join(selected))
class Checker:
    def check(self,answer,records,info):
        if not answer or not records:
            return False,0.0
        aw=set(w for w in alnum_words(answer))
        qw=set(info.get("important",[]))
        target=alnum_words(info.get("target",""))
        source=set()
        for r in records[:5]:
            source.update(alnum_words(r.get("t","")+" "+" ".join(r.get("a",[]))+" "+r.get("f","")))
        query_overlap=len(aw&qw)/max(1,len(qw))
        source_overlap=len(aw&source)/max(1,len(aw))
        target_overlap=len(set(target)&aw)/max(1,len(target)) if target else 0
        confidence=min(1.0,source_overlap*.45+query_overlap*.15+target_overlap*.40)
        if info["intent"]=="DEFINITION" and re.search(r"\b(is a|is an|refers to|means|defined as)\b",answer.lower()):
            confidence=min(1.0,confidence+.12)
        return confidence>=.52,round(confidence,3)
class AnswerCache:
    def __init__(self):
        self.data={}
    def load(self):
        if not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE,encoding="utf-8") as f:
                self.data=json.load(f)
        except:
            self.data={}
    def key(self,prompt):
        return compact(normalize(prompt))
    def get(self,prompt):
        return self.data.get(self.key(prompt))
    def put(self,prompt,result):
        self.data[self.key(prompt)]=result
        if len(self.data)>300:
            keys=list(self.data.keys())
            for k in keys[:len(self.data)-300]:
                del self.data[k]
        os.makedirs(os.path.dirname(CACHE_FILE),exist_ok=True)
        try:
            with open(CACHE_FILE,"w",encoding="utf-8") as f:
                json.dump(self.data,f,ensure_ascii=False,separators=(",",":"))
        except:
            pass
class Wikipedia:
    def search(self,q):
        words=alnum_words(q)
        if not words:
            return None
        query=" ".join(words)
        try:
            r=requests.get(WIKI_API,params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":10},timeout=10)
            results=r.json().get("query",{}).get("search",[])
            if not results:
                return None
            qcompact=compact(query)
            best=None
            bestscore=-1
            for x in results:
                title=normalize(x.get("title",""))
                tc=compact(title)
                score=0
                if tc==qcompact:
                    score+=1000
                elif qcompact in tc or tc in qcompact:
                    score+=300
                score+=len(set(words)&set(alnum_words(title)))*30
                if score>bestscore:
                    bestscore=score
                    best=title
            return best
        except:
            return None
    def get(self,title):
        try:
            r=requests.get(WIKI_API,params={"action":"query","prop":"extracts","explaintext":1,"titles":title,"format":"json","redirects":1},timeout=15)
            pages=r.json().get("query",{}).get("pages",{})
            for p in pages.values():
                if p.get("extract"):
                    return p["extract"]
        except:
            pass
        return ""
class GroqExtractor:
    def __init__(self):
        self.key=os.environ.get("GROQ_API_KEY")
    def extract(self,text,title):
        if not self.key:
            return []
        prompt=f'''Extract factual knowledge about "{title}".
Return ONLY JSON:
{{"records":[{{"topic":"","type":"","subjects":[],"aliases":[],"fact":""}}]}}
Rules:
- Only facts supported by the article.
- One main idea per fact.
- Keep facts concise.
- Put the main entity in topic.
- Include common names and abbreviations in aliases.
- For people use type "person".
- For definitions use type "definition".
- No jokes.
- No recommendations.
- No duplicate facts.
ARTICLE:
{text[:18000]}'''
        try:
            r=requests.post(GROQ_API,headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"},json={"model":GROQ_MODEL,"messages":[{"role":"system","content":"Return valid JSON only."},{"role":"user","content":prompt}],"temperature":.1,"max_tokens":3500,"response_format":{"type":"json_object"}},timeout=90)
            if r.status_code!=200:
                return []
            content=r.json().get("choices",[{}])[0].get("message",{}).get("content","")
            d=json.loads(content)
            out=[]
            seen=set()
            allowed={"definition","fact","cause","effect","process","example","history","person","location","time","formula"}
            for x in d.get("records",[]):
                if not isinstance(x,dict):
                    continue
                topic=normalize(x.get("topic",title))
                fact=normalize(x.get("fact",""))
                if not fact:
                    continue
                typ=clean(x.get("type","fact"))
                if typ not in allowed:
                    typ="fact"
                s=x.get("subjects",[])
                a=x.get("aliases",[])
                if not isinstance(s,list):
                    s=[]
                if not isinstance(a,list):
                    a=[]
                rec={"t":topic,"y":typ,"s":[normalize(str(z)) for z in s if str(z).strip()],"a":[normalize(str(z)) for z in a if str(z).strip()],"f":fact}
                k=compact(topic+"|"+fact)
                if k not in seen:
                    seen.add(k)
                    out.append(rec)
            return out
        except Exception as e:
            print("[Groq] Error:",e)
            return []
class Learner:
    def save(self,records):
        if not records:
            return False
        os.makedirs(DATA_DIR,exist_ok=True)
        keys=set()
        if os.path.exists(KNOWLEDGE):
            with open(KNOWLEDGE,encoding="utf-8") as f:
                for line in f:
                    try:
                        d=json.loads(line)
                        keys.add(compact(d.get("t","")+"|"+d.get("f","")))
                    except:
                        pass
        added=0
        with open(KNOWLEDGE,"a",encoding="utf-8") as f:
            for r in records:
                k=compact(r.get("t","")+"|"+r.get("f",""))
                if not k or k in keys:
                    continue
                f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
                keys.add(k)
                added+=1
        print("[ORIN] Added:",added,"new records")
        return added>0
class Orin:
    def __init__(self):
        self.model=WordModel()
        self.rag=RAG()
        self.semantic=FastSemantic384()
        self.semantic_store=SemanticStore()
        self.stitcher=Stitcher()
        self.checker=Checker()
        self.wiki=Wikipedia()
        self.groq=GroqExtractor()
        self.learner=Learner()
        self.cache=AnswerCache()
    def initialize(self):
        self.rag.load()
        self.model.load(WORDS)
        if self.semantic_store.load(SEMANTIC):
            print("[ORIN] Semantic vectors loaded:",f"{self.semantic_store.count:,}")
        self.cache.load()
        print("[ORIN] Initialization complete.")
    def prepare(self,prompt):
        info=analyze_sentence(prompt)
        if info["chat"]:
            return info
        raw=alnum_words(prompt)
        info["concepts"]=concepts(prompt)
        important=[w for w in raw if w not in STOP]
        # Build the actual entity phrase instead of taking only the last word.
        if info["question"] in {"who","where","when","what","which","why","how"}:
            qpos=-1
            for i,w in enumerate(raw):
                if w==info["question"]:
                    qpos=i
                    break
            if qpos>=0:
                tail=raw[qpos+1:]
                if tail and tail[0]=="is":
                    tail=tail[1:]
                if tail:
                    info["entity_phrase"]=" ".join(tail)
                else:
                    info["entity_phrase"]=""
        else:
            info["entity_phrase"]=""
        search_terms=[]
        entity=clean(info.get("entity_phrase",""))
        if entity:
            search_terms.append(entity)
        for c in info["concepts"]:
            search_terms.append(c)
        if not search_terms:
            search_terms.append(" ".join(important))
        info["search_terms"]=search_terms
        target=entity or (info["concepts"][0] if info["concepts"] else "")
        if not target and important:
            target=" ".join(important)
        # Expand common short names into canonical names.
        ct=compact(target)
        if ct in ALIASES:
            target=ALIASES[ct]
        elif clean(target) in ALIASES:
            target=ALIASES[clean(target)]
        info["target"]=target
        for c in info["concepts"]:
            for w in c.split():
                if w not in important:
                    important.append(w)
        info["important"]=important
        return info
    def chat_response(self,p):
        t=chat_clean(p)
        if t in {"hi","hello","hey","hiya"}:
            return "Hi. What do you want to know?"
        if t in {"how are you","how r you"}:
            return "I'm running normally. What do you want to know?"
        if t in {"who are you","what are you"}:
            return "I'm Orin, a local symbolic and semantic knowledge system."
        if t in {"thanks","thank you","thx"}:
            return "You're welcome."
        if t in {"bye","goodbye"}:
            return "Goodbye."
        if t in {"whats up","what's up","wassup"}:
            return "I'm here and ready. What do you want to know?"
        return None
    def learn(self,info):
        q=info.get("target") or " ".join(info["important"])
        print("[ORIN] Local knowledge insufficient.")
        print("[ORIN] Searching Wikipedia...")
        title=self.wiki.search(q)
        if not title:
            return []
        print("[ORIN] Wikipedia:",title)
        article=self.wiki.get(title)
        if not article:
            return []
        records=self.groq.extract(article,title)
        if self.learner.save(records):
            self.rag.build()
            self.build_semantic()
            self.semantic_store.load(SEMANTIC)
        return records
    def ask(self,prompt):
        chat=self.chat_response(prompt)
        if chat:
            print("\n[ORIN]",chat,"\n")
            return
        cached=self.cache.get(prompt)
        if cached:
            print("\n[ORIN] "+cached+"\n")
            return
        info=self.prepare(prompt)
        records=self.rag.search(info,self.semantic,self.semantic_store,10)
        answer=self.stitcher.choose(records,info)
        grounded,confidence=self.checker.check(answer,records,info) if answer else (False,0.0)
        source="LOCAL"
        good=bool(answer and grounded)
        # For definitions, weak generic sentences are not acceptable.
        if info["intent"]=="DEFINITION" and confidence<.60:
            good=False
        if not good:
            learned=self.learn(info)
            if learned:
                records=self.rag.search(info,self.semantic,self.semantic_store,10)
                answer=self.stitcher.choose(records,info)
                grounded,confidence=self.checker.check(answer,records,info) if answer else (False,0.0)
                source="WIKIPEDIA + GROQ"
        if not answer or not grounded:
            answer="I could not find enough reliable information."
            source="NONE"
            grounded=False
            confidence=0.0
        result=(answer,source,grounded,confidence)
        self.cache.put(prompt,result)
        print("\n"+"="*65)
        print("ORIN v18")
        print("="*65)
        print("Sentence Type :",info["type"])
        print("Intent        :",info["intent"])
        print("Concepts      :",", ".join(info.get("concepts",[])) or "none")
        print("Important     :",", ".join(info.get("important",[])) or "none")
        print("Target        :",info.get("target") or "none")
        print("Candidates    :",len(self.rag.candidates(info)))
        print("Source        :",source)
        print("Grounded      :",grounded)
        print("Confidence    :",confidence)
        print("-"*65)
        print(answer)
        print("="*65+"\n")
    def train(self):
        print("\n"+"="*65)
        print("ORIN v18 COMPILATION")
        print("="*65)
        print("[1/3] Compiling vocabulary...")
        self.model.train(TRAIN)
        self.model.save(WORDS)
        print("[2/3] Building symbolic RAG...")
        self.rag.build()
        print("[3/3] Building semantic index...")
        self.build_semantic()
        print("="*65)
        print("ORIN v18 COMPILATION COMPLETE")
        print("="*65+"\n")
    def build_semantic(self):
        if not self.rag.records:
            self.rag.load_knowledge()
        vectors=[]
        total=len(self.rag.records)
        for i,r in enumerate(self.rag.records):
            text=r.get("t","")+" "+" ".join(r.get("s",[]))+" "+" ".join(r.get("a",[]))+" "+r.get("f","")
            vectors.append(self.semantic.vectorize(text))
            if (i+1)%500==0:
                print("[SEMANTIC]",i+1,"/",total)
        SemanticStore().save(SEMANTIC,vectors)
        print("[SEMANTIC] Saved:",SEMANTIC)
        print("[SEMANTIC] Dimensions:",DIM)
        print("[SEMANTIC] Vectors:",len(vectors))
        print("[SEMANTIC] Storage:",f"{os.path.getsize(SEMANTIC)/1024/1024:.2f} MB")
    def chat(self):
        self.initialize()
        print("\n"+"="*65)
        print("ORIN v18 CHAT")
        print("="*65)
        print("Hybrid Symbolic + Fast Pure Python Semantic RAG")
        print("Exact entity + definition priority")
        print("Persistent answer cache")
        print("384-dimensional cached semantic vectors")
        print("Binary float32 semantic index")
        print("No sentence-transformers")
        print("No NumPy")
        print("No PyTorch")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                p=input("You: ").strip()
            except (KeyboardInterrupt,EOFError):
                print("\n[ORIN] Goodbye.")
                break
            if not p:
                continue
            if chat_clean(p) in {"exit","quit"}:
                print("[ORIN] Goodbye.")
                break
            self.ask(p)
def main():
    o=Orin()
    if len(sys.argv)>1:
        if sys.argv[1]=="--train":
            o.train()
            return
        if sys.argv[1]=="--ask" and len(sys.argv)>2:
            o.initialize()
            o.ask(" ".join(sys.argv[2:]))
            return
        if sys.argv[1]=="--build-semantic":
            o.rag.load()
            o.build_semantic()
            return
    o.chat()
if __name__=="__main__":
    main()
