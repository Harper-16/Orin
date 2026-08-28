import os,re,json,urllib.parse,urllib.request
import numpy as np
BASE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(BASE,"data")
KNOW=os.path.join(DATA,"knowledge.jsonl")
LEARN=os.path.join(DATA,"learned.jsonl")
SEM=os.path.join(DATA,"train","semantic_index.npz")
LSEM=os.path.join(DATA,"train","learned_vectors.npz")
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None
STOP=set("""a an the is are was were be been being what who where when why how do does did can could would should will i me my you your he she it they them we our and or of to in on at for from with about between this that these those tell explain give show please""".split())
ALIASES={
    "ai":"artificial intelligence",
    "a.i":"artificial intelligence",
    "a.i.":"artificial intelligence",
    "artificial intelligence":"artificial intelligence",
    "math":"mathematics",
    "maths":"mathematics",
    "mathematical science":"mathematics",
    "cs":"computer science",
    "comp sci":"computer science",
    "computer science":"computer science",
    "physic":"physics",
    "physics":"physics",
    "bio":"biology",
    "chem":"chemistry",
    "nlp":"natural language processing",
    "spiderman":"spider-man",
    "spider man":"spider-man",
    "open ai":"openai"
}
GREET={"hi","hello","hey","hiya","yo","good morning","good afternoon","good evening"}
BYE={"bye","goodbye","good bye","see you","exit","quit"}
def clean(s):
    s=str(s or "").lower().replace("\n"," ").replace("\t"," ")
    s=re.sub(r"\[[^\]]*\]"," ",s)
    s=re.sub(r"\s+"," ",s)
    return s.strip(" .,:;!?")
def words(s):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?",clean(s))
def norm_word(w):
    w=w.lower()
    if len(w)<=3:
        return w
    if len(w)>5 and w.endswith("ies"):
        return w[:-3]+"y"
    if len(w)>5 and w.endswith("ing"):
        return w[:-3]
    if len(w)>4 and w.endswith("ed"):
        return w[:-2]
    if len(w)>4 and w.endswith("es"):
        return w[:-2]
    if len(w)>3 and w.endswith("s"):
        return w[:-1]
    return w
def normalize(s):
    return clean(s)
def alias(x):
    x=clean(x)
    if x in ALIASES:
        return ALIASES[x]
    for a,b in sorted(ALIASES.items(),key=lambda x:-len(x[0])):
        x=re.sub(r"\b"+re.escape(a)+r"\b",b,x)
    return clean(x)
def cosine(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d else 0.0
def token_score(a,b):
    A=set(norm_word(x) for x in words(a) if x not in STOP)
    B=set(norm_word(x) for x in words(b) if x not in STOP)
    return len(A&B)/len(A) if A and B else 0.0
def overlap(a,b):
    A=set(norm_word(x) for x in words(a))
    B=set(norm_word(x) for x in words(b))
    return len(A&B)/max(1,len(A))
def detect_intent(q):
    w=words(normalize(q))
    if not w:
        return "STATEMENT"
    if w[0]=="who":
        return "PERSON"
    if w[0]=="where":
        return "LOCATION"
    if w[0]=="why":
        return "WHY"
    if w[0]=="how":
        return "HOW"
    if w[0] in {"what","define","definition","meaning","explain","describe"}:
        return "DEFINITION"
    return "DEFINITION" if "what" in w else "STATEMENT"
def target(q,intent):
    x=normalize(q)
    patterns={
        "DEFINITION":[
            r"^\s*what\s+(?:is|are|does)\s+(?:(?:a|an|the)\s+)?(.+?)(?:\?|$)",
            r"^\s*(?:define|meaning\s+of|explain|describe)\s+(?:(?:a|an|the)\s+)?(.+?)(?:\?|$)"
        ],
        "PERSON":[
            r"^\s*who\s+(?:is|was|are|were)?\s*(.+?)(?:\?|$)"
        ],
        "LOCATION":[
            r"^\s*where\s+(?:is|was|are|were)?\s*(.+?)(?:\?|$)"
        ],
        "HOW":[
            r"^\s*how\s+(?:do|does|can|to)?\s*(.+?)(?:\?|$)"
        ],
        "WHY":[
            r"^\s*why\s+(?:is|are|was|were|does|do)?\s*(.+?)(?:\?|$)"
        ]
    }
    if intent in patterns:
        for p in patterns[intent]:
            m=re.search(p,x)
            if m:
                x=m.group(1).strip()
                break
    else:
        x=" ".join(z for z in words(x) if z not in STOP)
    x=clean(x)
    x=alias(x)
    return x
def concepts(q):
    x=normalize(q)
    t=target(q,detect_intent(q))
    out=[t] if t else []
    for a,b in sorted(ALIASES.items(),key=lambda z:-len(z[0])):
        if re.search(r"\b"+re.escape(a)+r"\b",x) and b not in out:
            out.append(b)
    return out
def sentence_type(q):
    q=normalize(q)
    return "INTERROGATIVE" if q.endswith("?") or words(q)[:1] in [["what"],["who"],["where"],["when"],["why"],["how"]] else "DECLARATIVE"
class Knowledge:
    def __init__(self):
        self.rows=[]
        self.learned=[]
        self.load_rows(KNOW,self.rows)
        self.load_rows(LEARN,self.learned)
        self.base_texts=[self.text(r) for r in self.rows]
        self.learn_texts=[self.text(r) for r in self.learned]
        self.texts=self.base_texts+self.learn_texts
        self.index={}
        for i,t in enumerate(self.texts):
            for w in set(norm_word(x) for x in words(t)):
                if len(w)>1:
                    self.index.setdefault(w,[]).append(i)
        self.vec=None
        self.lvec=None
        self.model=None
        self.semantic_ok=False
        self.load_semantic()
        print("[ORIN] Records:",format(len(self.rows),","))
        print("[ORIN] Learned:",format(len(self.learned),","))
        print("[ORIN] Index words:",format(len(self.index),","))
        if self.semantic_ok:
            print("[ORIN] Semantic index:",self.vec.shape)
            if self.lvec is not None:
                print("[ORIN] Learned vectors:",self.lvec.shape)
        else:
            print("[ORIN] Semantic index unavailable. Symbolic mode.")
    def load_rows(self,path,out):
        if not os.path.exists(path):
            return
        try:
            with open(path,"r",encoding="utf-8") as f:
                for line in f:
                    try:
                        r=json.loads(line)
                        if isinstance(r,dict):
                            out.append(r)
                    except Exception:
                        pass
        except Exception as e:
            print("[ORIN] Load error:",e)
    def text(self,r):
        fields=[]
        for k in ("knowledge_concept","concept","topic","subtopic1","subtopic2","subtopic3","subject_hierarchy","atomic_factual_statement","fact","text","content"):
            if r.get(k):
                fields.append(str(r[k]))
        return clean(". ".join(fields))
    def load_semantic(self):
        try:
            if not os.path.exists(SEM) or not SentenceTransformer:
                return
            z=np.load(SEM)
            key="vectors" if "vectors" in z else ("embeddings" if "embeddings" in z else z.files[0])
            self.vec=z[key].astype(np.float32)
            if len(self.vec)!=len(self.rows):
                print("[ORIN] Base semantic count mismatch.")
                self.vec=None
                return
            self.vec/=np.maximum(np.linalg.norm(self.vec,axis=1,keepdims=True),1e-12)
            if os.path.exists(LSEM):
                z=np.load(LSEM)
                key="vectors" if "vectors" in z else z.files[0]
                self.lvec=z[key].astype(np.float32)
                if len(self.lvec)!=len(self.learned):
                    print("[ORIN] Learned semantic count mismatch.")
                    self.lvec=None
            self.model=SentenceTransformer("all-MiniLM-L6-v2")
            self.semantic_ok=True
        except Exception as e:
            print("[ORIN] Semantic load error:",e)
    def symbolic(self,q,lim=30):
        q=normalize(q)
        ws=set(norm_word(x) for x in words(q) if x not in STOP)
        ids=set()
        for w in ws:
            ids.update(self.index.get(w,[]))
        scored=[]
        for i in ids:
            s=token_score(q,self.texts[i])
            if s>0:
                scored.append((s,i))
        return sorted(scored,reverse=True)[:lim]
    def semantic(self,q,lim=30):
        if not self.semantic_ok:
            return []
        try:
            v=self.model.encode([normalize(q)],normalize_embeddings=True,show_progress_bar=False)[0].astype(np.float32)
            out=[]
            if self.vec is not None:
                s=self.vec@v
                n=min(lim,len(s))
                if n:
                    ids=np.argpartition(-s,n-1)[:n]
                    out += [(float(s[i]),int(i)) for i in ids]
            if self.lvec is not None:
                s=self.lvec@v
                n=min(lim,len(s))
                if n:
                    ids=np.argpartition(-s,n-1)[:n]
                    out += [(float(s[i]),len(self.rows)+int(i)) for i in ids]
            return sorted(out,reverse=True)[:lim*2]
        except Exception as e:
            print("[ORIN] Semantic search error:",e)
            return []
    def rerank(self,q,intent):
        tar=target(q,intent)
        sy=self.symbolic(q,50)
        se=self.semantic(q,50)
        ids={i for _,i in sy}|{i for _,i in se}
        sd=dict(sy)
        vd=dict(se)
        result=[]
        for i in ids:
            text=self.texts[i]
            sym=sd.get(i,0)
            sem=vd.get(i,0)
            tok=token_score(tar,text)
            ov=overlap(tar,text)
            exact=1.0 if tar and re.search(r"\b"+re.escape(tar)+r"\b",text) else 0.0
            score=.20*sem+.18*sym+.27*tok+.25*exact+.10*ov
            if intent=="DEFINITION":
                if tar and re.search(r"\b"+re.escape(tar)+r"\s+(?:is|are|refers to|means)\b",text):
                    score+=.50
                elif tar and re.search(r"\b"+re.escape(tar)+r"\b",text):
                    score+=.05
                if re.search(r"\bwhat\s+(?:is|are)\b",text):
                    score-=.35
                if re.search(r"\bq\.\s|\ba\.\s|question|answer",text):
                    score-=.45
            elif intent=="PERSON":
                if tar and tar in text:
                    score+=.35
                if re.search(r"\b(born|died|actor|physicist|scientist|mathematician|politician|artist|writer)\b",text):
                    score+=.12
            elif intent=="LOCATION":
                if tar and tar in text:
                    score+=.30
            elif intent=="WHY":
                if re.search(r"\b(because|due to|reason|caused)\b",text):
                    score+=.18
            elif intent=="HOW":
                if re.search(r"\b(process|method|steps|procedure|works|used)\b",text):
                    score+=.12
            if len(text)<35:
                score-=.25
            result.append((score,i))
        return sorted(result,reverse=True)
    def answer(self,q):
        intent=detect_intent(q)
        tar=target(q,intent)
        ranked=self.rerank(q,intent)
        if not ranked:
            return None,0,0
        best,i=ranked[0]
        text=self.texts[i]
        if intent=="DEFINITION":
            direct=bool(tar and re.search(r"\b"+re.escape(tar)+r"\s+(?:is|are|refers to|means)\b",text))
            if not direct and best<.62:
                return None,0,len(ranked)
        elif intent in {"PERSON","LOCATION"} and best<.55:
            return None,0,len(ranked)
        elif best<.48:
            return None,0,len(ranked)
        return self.clean_answer(text,tar,intent),min(1,max(0,best)),len(ranked)
    def clean_answer(self,text,tar,intent):
        text=re.sub(r"\s+"," ",text).strip()
        text=text.replace("bhe ","the ").replace("..",".")
        if intent=="DEFINITION" and tar:
            m=re.search(r"\b"+re.escape(tar)+r"\s+(?:is|are|refers to|means)\b.*?(?:\.\s|$)",text)
            if m and len(m.group(0))>=35:
                text=m.group(0).strip()
        if len(text)>550:
            shortened=text[:550].rsplit(".",1)[0]
            if shortened:
                text=shortened+"."
        return text
    def learn(self,target,text):
        target=alias(clean(target))
        text=clean(text)
        if not target or not text or len(text)<25:
            return False
        for r in self.learned:
            if clean(r.get("knowledge_concept",""))==target and clean(r.get("fact",""))==text:
                return False
        r={"knowledge_concept":target,"fact":text,"source":"wikipedia"}
        try:
            with open(LEARN,"a",encoding="utf-8") as f:
                f.write(json.dumps(r,ensure_ascii=False)+"\n")
            self.learned.append(r)
            self.learn_texts.append(text)
            self.texts.append(text)
            i=len(self.texts)-1
            for w in set(norm_word(x) for x in words(text)):
                if len(w)>1:
                    self.index.setdefault(w,[]).append(i)
            if self.semantic_ok:
                v=self.model.encode([text],normalize_embeddings=True,show_progress_bar=False)[0].astype(np.float32)
                if self.lvec is None:
                    self.lvec=np.empty((0,len(v)),dtype=np.float32)
                self.lvec=np.vstack([self.lvec,v])
                os.makedirs(os.path.dirname(LSEM),exist_ok=True)
                np.savez_compressed(LSEM,vectors=self.lvec)
            print("[ORIN] Learned:",target)
            return True
        except Exception as e:
            print("[ORIN] Learning error:",e)
            return False
class Wikipedia:
    def search(self,q):
        candidates=[]
        q=clean(q)
        if q:
            candidates.append(q)
        a=alias(q)
        if a and a not in candidates:
            candidates.append(a)
        if q.endswith("s") and len(q)>3:
            candidates.append(q[:-1])
        for x in candidates:
            try:
                u="https://en.wikipedia.org/api/rest_v1/page/summary/"+urllib.parse.quote(x.replace(" ","_"),safe="")
                req=urllib.request.Request(u,headers={"User-Agent":"Orin/1.0"})
                with urllib.request.urlopen(req,timeout=8) as r:
                    d=json.loads(r.read().decode())
                if d.get("type")=="disambiguation":
                    continue
                text=d.get("extract")
                if text:
                    return clean(text)
            except Exception:
                pass
        return None
class Orin:
    def __init__(self):
        self.k=Knowledge()
        self.wiki=Wikipedia()
        print("[ORIN] Initialization complete.")
    def chat(self,q):
        raw=clean(q)
        if not raw:
            return ""
        low=normalize(raw)
        if low in GREET:
            return "[ORIN] Hi. What do you want to know?"
        if low in BYE:
            return "[ORIN] Goodbye."
        if low in {"what are you","who are you","what is orin","who is orin"}:
            return "[ORIN] I'm Orin, a local meaning-aware knowledge system."
        if low in {"how are you","how are u"}:
            return "[ORIN] I'm running normally. What do you want to know?"
        if low in {"what s up","whats up","what is up","sup"}:
            return "[ORIN] Not much. I'm ready to answer questions."
        intent=detect_intent(raw)
        tar=target(raw,intent)
        cs=concepts(raw)
        ans,conf,cands=self.k.answer(raw)
        if ans:
            return self.format(raw,intent,cs,tar,ans,cands,conf,"LOCAL",True)
        print("[ORIN] Local knowledge insufficient.")
        print("[ORIN] Searching Wikipedia:",tar or raw)
        w=self.wiki.search(tar or raw)
        if w:
            self.k.learn(tar or raw,w)
            return self.format(raw,intent,cs,tar,w,0,.75,"WIKIPEDIA",True)
        return self.format(raw,intent,cs,tar,"I could not find enough reliable information.",0,0,"NONE",False)
    def format(self,q,intent,cs,tar,ans,cands,conf,source,grounded):
        print("\n"+"="*65)
        print("ORIN v4")
        print("="*65)
        print("Sentence Type :",sentence_type(q))
        print("Intent        :",intent)
        print("Concepts      :",", ".join(cs) if cs else "none")
        print("Important     :",", ".join(norm_word(x) for x in words(q) if x not in STOP) or "none")
        print("Target        :",tar or "none")
        print("Candidates    :",cands)
        print("Source        :",source)
        print("Grounded      :",grounded)
        print("Confidence    :",round(conf,3))
        print("-"*65)
        print(ans)
        print("="*65)
        return ans
def main():
    print("\n"+"="*65)
    print("ORIN v4 CHAT")
    print("="*65)
    print("Hybrid Semantic + Symbolic RAG + Persistent Learning")
    print("Type 'exit' to quit.")
    print()
    o=Orin()
    while True:
        try:
            q=clean(input("You: "))
        except (KeyboardInterrupt,EOFError):
            print("\n[ORIN] Goodbye.")
            break
        if not q:
            continue
        try:
            a=o.chat(q)
            print("\n"+a+"\n")
            if normalize(q) in BYE:
                break
        except Exception as e:
            print("[ORIN] Error:",e)
if __name__=="__main__":
    main()