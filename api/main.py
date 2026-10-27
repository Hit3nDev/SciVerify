"""FastAPI backend and web app.

    uvicorn api.main:app --port 8000        then open http://localhost:8000

    GET  /                 the web app (frontend/web)
    GET  /judges           the NLI judges the app can switch between (configs/verification.yaml)
    POST /verify           {"claim": "...", "top_k"?: 1-10, "min_confidence"?: 0-1, "model"?: judge key}
    POST /verify/stream    same body; newline-delimited JSON events as each stage finishes
    POST /counterclaim     same body; verifies the claim AND its opposite and compares the verdicts
    POST /explain          {"claim": "...", "doc_id": n, "model"?}  ->  a verdict for every sentence of one abstract
    GET  /search?q=&k=     retrieval only, no model
    GET  /papers/{doc_id}  one abstract
    GET  /examples         random labelled dev claims (with the dataset's answer)
    GET  /benchmarks       the recorded evaluation results in experiments/results
    GET  /status, POST /warmup?model=, GET /health

Each judge (corpus + NLI model) loads in the background when the page first calls /warmup, or
on the first request that needs it, so the server itself starts instantly.
"""

from __future__ import annotations

import inspect
import json
import random
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from verification.counterclaim import build_counterclaim, judge_consistency

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "frontend" / "web"
RESULTS_DIR = ROOT / "experiments" / "results"
CACHE_SIZE = 256


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=5000, description="A scientific claim in plain English.")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Papers to judge (default from config).")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0,
                                            description="NLI probability a paper needs to count.")
    model: Optional[str] = Field(None, description="Judge key from GET /judges (default: the first one).")


class ExplainRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=5000)
    doc_id: int
    model: Optional[str] = None


class _LazyVerifier:
    """Builds one verifier once, either in a background thread (warm-up) or on first use."""

    def __init__(self, factory: Callable[[], object]) -> None:
        self._factory, self._lock = factory, threading.Lock()
        self._value, self.error, self.loading = None, None, False

    @property
    def ready(self) -> bool:
        return self._value is not None

    def get(self):
        with self._lock:
            if self._value is None:
                self.loading = True
                try:
                    self._value, self.error = self._factory(), None
                except Exception as exc:  # noqa: BLE001 -- surfaced through /status
                    self.error = str(exc)
                    raise
                finally:
                    self.loading = False
            return self._value

    def warm_up(self) -> None:
        if self._value is None and not self.loading:
            self.loading = True                       # set before the thread starts so /status sees it
            threading.Thread(target=self._quiet_get, daemon=True).start()

    def _quiet_get(self) -> None:
        try:
            self.get()
        except Exception:  # noqa: BLE001
            pass


def _load_results(results_dir: Path) -> dict:
    out = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def create_app(verifier_factory: Optional[Callable] = None,
               claims_factory: Optional[Callable[[], List[dict]]] = None,
               describe: Optional[Callable[[], dict]] = None,
               results_dir: Path = RESULTS_DIR,
               judges: Optional[List[dict]] = None) -> FastAPI:
    """`verifier_factory` is either a zero-argument callable (one judge) or takes a judge key
    (one verifier per entry of `judges`). Without it the real VerifierPool is used."""
    app = FastAPI(title="SciVerify", version="0.3.0",
                  description="Checks a scientific claim against retrieved paper abstracts.")

    if verifier_factory is None:
        from verification.factory import VerifierPool
        pool = VerifierPool()
        verifier_factory, judges = pool.get, pool.judges
    if judges is None:
        judges = [{"key": "default", "path": "", "name": "default judge", "description": "", "calibrated": False}]
    takes_key = len(inspect.signature(verifier_factory).parameters) >= 1
    if claims_factory is None:
        from verification.factory import load_labelled_claims
        claims_factory = load_labelled_claims
    if describe is None:
        from verification.factory import describe

    default_key = judges[0]["key"]
    verifiers: Dict[str, _LazyVerifier] = {
        j["key"]: _LazyVerifier((lambda k=j["key"]: verifier_factory(k)) if takes_key else verifier_factory)
        for j in judges}
    cache: "OrderedDict[tuple, dict]" = OrderedDict()
    claims_state: dict = {}

    def resolve(model: Optional[str]) -> str:
        key = model or default_key
        if key not in verifiers:
            raise HTTPException(status_code=422, detail=f"Unknown judge '{model}'. See GET /judges.")
        return key

    def any_ready_verifier():
        """For corpus-only work (search, reading a paper) any loaded judge will do."""
        for lv in verifiers.values():
            if lv.ready:
                return lv.get()
        return verifiers[default_key].get()

    def judge_list() -> List[dict]:
        return [{**j, "ready": verifiers[j["key"]].ready, "loading": verifiers[j["key"]].loading,
                 "error": verifiers[j["key"]].error} for j in judges]

    def labelled_claims() -> List[dict]:
        if "claims" not in claims_state:
            try:
                claims_state["claims"] = claims_factory()
            except Exception:  # noqa: BLE001 -- dataset missing: the feature just has no data
                claims_state["claims"] = []
        return claims_state["claims"]

    def gold_for(claim: str) -> Optional[dict]:
        text = claim.strip()
        for c in labelled_claims():
            if c["claim"] == text:
                return {"gold_verdict": c["gold_verdict"], "gold_doc_ids": c["gold_doc_ids"]}
        return None

    def cache_key(req: VerifyRequest, key: str) -> tuple:
        return (req.claim.strip(), req.top_k, req.min_confidence, key)

    def remember(ck: tuple, result: dict) -> None:
        cache[ck] = result
        cache.move_to_end(ck)
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)

    def run_verify(req: VerifyRequest) -> dict:
        key = resolve(req.model)
        ck = cache_key(req, key)
        if ck in cache:
            cache.move_to_end(ck)
            return {**cache[ck], "cached": True}
        try:
            result = verifiers[key].get().verify(req.claim, top_k=req.top_k,
                                                 min_confidence=req.min_confidence).to_dict()
        except ValueError as exc:                     # e.g. whitespace-only claim
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["gold"] = gold_for(req.claim)
        result["judge_key"] = key
        remember(ck, result)
        return {**result, "cached": False}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/judges")
    def get_judges() -> dict:
        return {"default": default_key, "judges": judge_list()}

    @app.get("/status")
    def status() -> dict:
        try:
            info = describe()
        except Exception:  # noqa: BLE001
            info = {}
        default = verifiers[default_key]
        corpus_size = len(default.get().corpus) if default.ready else None
        return {"model_ready": default.ready, "loading": default.loading, "error": default.error,
                "corpus_size": corpus_size, "cached_results": len(cache), "judges": judge_list(), **info}

    @app.post("/warmup")
    def warmup(model: Optional[str] = None) -> dict:
        key = resolve(model)
        verifiers[key].warm_up()
        return {"judge": key, "model_ready": verifiers[key].ready, "loading": verifiers[key].loading}

    @app.post("/verify")
    def verify(req: VerifyRequest) -> dict:
        return run_verify(req)

    @app.post("/verify/stream")
    def verify_stream(req: VerifyRequest) -> StreamingResponse:
        if not req.claim.strip():
            raise HTTPException(status_code=422, detail="The claim is empty.")
        key = resolve(req.model)
        ck = cache_key(req, key)

        def line(obj: dict) -> str:
            return json.dumps(obj) + "\n"

        def events():
            if ck in cache:
                result = cache[ck]
                yield line({"event": "retrieved", "papers": [
                    {"doc_id": e["doc_id"], "title": e["title"], "retrieval_score": e["retrieval_score"]}
                    for e in result["evidence"]]})
                for e in result["evidence"]:
                    yield line({"event": "judged", "item": e})
                yield line({"event": "done", "result": {**result, "cached": True}})
                return
            try:
                lv = verifiers[key]
                if not lv.ready:
                    yield line({"event": "loading"})
                v = lv.get()
                yield line({"event": "started"})
                for kind, payload in v.verify_steps(req.claim, top_k=req.top_k,
                                                    min_confidence=req.min_confidence):
                    if kind == "retrieved":
                        yield line({"event": "retrieved", "papers": payload})
                    elif kind == "judged":
                        yield line({"event": "judged", "item": asdict(payload)})
                    else:
                        result = payload.to_dict()
                        result["gold"] = gold_for(req.claim)
                        result["judge_key"] = key
                        remember(ck, result)
                        yield line({"event": "done", "result": {**result, "cached": False}})
            except Exception as exc:  # noqa: BLE001 -- the stream has started; report, don't crash
                yield line({"event": "error", "detail": str(exc)})

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/counterclaim")
    def counterclaim(req: VerifyRequest) -> dict:
        try:
            cc = build_counterclaim(req.claim)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        original = run_verify(req)
        counter = run_verify(req.model_copy(update={"claim": cc.text}))
        status_, note = judge_consistency(original["verdict"], counter["verdict"])
        return {"claim": original["claim"], "verdict": original["verdict"], "confidence": original["confidence"],
                "counter_claim": cc.text, "method": cc.method, "changed": list(cc.changed) if cc.changed else None,
                "counter_verdict": counter["verdict"], "counter_confidence": counter["confidence"],
                "counter_explanation": counter["explanation"],
                "counter_evidence": [{"doc_id": e["doc_id"], "title": e["title"], "judgement": e["judgement"],
                                      "nli_score": e["nli_score"]} for e in counter["evidence"]],
                "status": status_, "note": note, "judge_key": original.get("judge_key")}

    @app.post("/explain")
    def explain(req: ExplainRequest) -> dict:
        try:
            sentences = verifiers[resolve(req.model)].get().explain_sentences(req.claim, req.doc_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"No paper with id {req.doc_id}.") from exc
        return {"doc_id": req.doc_id, "sentences": sentences}

    @app.get("/search")
    def search(q: str = Query(..., min_length=1, max_length=1000), k: int = Query(10, ge=1, le=50)) -> dict:
        return {"query": q, "results": any_ready_verifier().search(q, top_k=k)}

    @app.get("/papers/{doc_id}")
    def paper(doc_id: int) -> dict:
        doc = any_ready_verifier().corpus.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No paper with id {doc_id}.")
        return {"doc_id": doc.doc_id, "title": doc.title, "sentences": list(doc.abstract)}

    @app.get("/examples")
    def examples(n: int = Query(1, ge=1, le=50), label: Optional[str] = None,
                 seed: Optional[int] = None) -> dict:
        pool_ = labelled_claims()
        if label:
            pool_ = [c for c in pool_ if c["gold_verdict"] == label.upper()]
        rng = random.Random(seed)
        return {"claims": rng.sample(pool_, min(n, len(pool_))), "pool_size": len(pool_)}

    @app.get("/benchmarks")
    def benchmarks() -> dict:
        return _load_results(results_dir)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
