"""
05_eval_and_merge.py - measure the fine-tune on the held-out val set, then (optionally) merge the adapter.

Compares BASE vs FINE-TUNED on the *same* loaded weights (adapter toggled off/on), reporting the things
that actually matter for the RAG app:

  answer task : refusal agreement, refusal recall, citation validity, format compliance, hedge-word rate
  route task  : valid JSON rate, intent accuracy, source accuracy

Usage:
    python finetune/05_eval_and_merge.py --model <BASE> --adapter finetune/outputs/legal-lora/final
    python finetune/05_eval_and_merge.py --model <BASE> --adapter <ADAPTER> --merge-to finetune/outputs/legal-lora/merged
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import add_mm_ids, chat_prompt_text, load_model, load_tokenizer, probe_text_only  # noqa: E402

FT_DIR = Path(__file__).resolve().parent
INSUFFICIENT_SENTENCE = "The available legal evidence is insufficient to answer this question."
CITE_RE = re.compile(r"\[E(\d+)\]")
HEDGE_RE = re.compile(r"\b(implied|inferred|presumably|i think|as an ai|it seems that|probably)\b", re.I)


def load_rows(path, n_answer, n_route):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ans = [r for r in rows if r["task"] == "answer"][:n_answer]
    rte = [r for r in rows if r["task"] == "route"][:n_route]
    return ans + rte


@torch.no_grad()
def generate(model, tok, rows, batch_size, max_new_tokens):
    tok.padding_side = "left"
    outs = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["messages"][0]["content"]))  # less padding
    for s in tqdm(range(0, len(order), batch_size), desc="generate"):
        idx = order[s:s + batch_size]
        prompts = [chat_prompt_text(tok, rows[i]["messages"][0]["content"]) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        enc = add_mm_ids(dict(enc))
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return outs


def is_refusal(t):
    return t.strip().startswith(INSUFFICIENT_SENTENCE)


def score(rows, preds):
    ans = [(r, p) for r, p in zip(rows, preds) if r["task"] == "answer"]
    rte = [(r, p) for r, p in zip(rows, preds) if r["task"] == "route"]
    m = {}
    if ans:
        m["refusal_agreement"] = sum(is_refusal(p) == r["refusal"] for r, p in ans) / len(ans)
        gold_ref = [(r, p) for r, p in ans if r["refusal"]]
        gold_ok = [(r, p) for r, p in ans if not r["refusal"]]
        if gold_ref:
            m["refusal_recall"] = sum(is_refusal(p) for _, p in gold_ref) / len(gold_ref)
        if gold_ok:
            def cite_ok(r, p):
                c = [int(x) for x in CITE_RE.findall(p)]
                return bool(c) and all(1 <= x <= r["n_evidence"] for x in c)
            m["citation_valid"] = sum(cite_ok(r, p) for r, p in gold_ok) / len(gold_ok)
            m["format_ok"] = sum("**Short answer" in p for _, p in gold_ok) / len(gold_ok)
            m["false_refusal_rate"] = sum(is_refusal(p) for _, p in gold_ok) / len(gold_ok)
        m["hedge_rate"] = sum(bool(HEDGE_RE.search(p)) for _, p in ans) / len(ans)
        m["mean_words"] = sum(len(p.split()) for _, p in ans) / len(ans)
    if rte:
        valid = ia = sa = 0
        for r, p in rte:
            try:
                got = json.loads(re.sub(r"^```[a-z]*\n?|\n?```$", "", p.strip()))
                gold = json.loads(r["messages"][1]["content"])
                valid += 1
                ia += got.get("intent", "").upper() == gold["intent"]
                sa += got.get("source", "").upper() == gold["source"]
            except Exception:  # noqa: BLE001
                pass
        m["route_json_valid"] = valid / len(rte)
        m["route_intent_acc"] = ia / len(rte)
        m["route_source_acc"] = sa / len(rte)
    return {k: round(v, 3) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BASE model path")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (e.g. outputs/legal-lora/final)")
    ap.add_argument("--val", default=str(FT_DIR / "data" / "val.jsonl"))
    ap.add_argument("--n-answer", type=int, default=120)
    ap.add_argument("--n-route", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--skip-base", action="store_true", help="don't evaluate the un-tuned baseline")
    ap.add_argument("--show", type=int, default=3, help="print N sample generations")
    ap.add_argument("--merge-to", default=None, help="merge adapter into the base weights and save here")
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    model, arch = load_model(args.model, args.quant)
    if arch["is_vl"]:
        probe_text_only(model, tok)

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = load_rows(args.val, args.n_answer, args.n_route)
    print(f"Evaluating on {len(rows)} held-out examples")
    report, tuned_preds = {}, None

    if args.adapter:
        tuned_preds = generate(model, tok, rows, args.batch_size, args.max_new_tokens)
        report["tuned"] = score(rows, tuned_preds)
        if not args.skip_base:
            with model.disable_adapter():
                base_preds = generate(model, tok, rows, args.batch_size, args.max_new_tokens)
            report["base"] = score(rows, base_preds)
    else:
        report["base"] = score(rows, generate(model, tok, rows, args.batch_size, args.max_new_tokens))

    keys = sorted({k for v in report.values() for k in v})
    print(f"\n{'metric':<22}" + "".join(f"{t:>10}" for t in report))
    for k in keys:
        print(f"{k:<22}" + "".join(f"{report[t].get(k, float('nan')):>10}" for t in report))

    out_json = FT_DIR / "outputs" / "eval_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if tuned_preds and args.show:
        print("\n" + "=" * 80 + "\nSAMPLE (tuned) GENERATIONS")
        for r, p in list(zip(rows, tuned_preds))[: args.show]:
            q = r["messages"][0]["content"].split("USER QUESTION:")[-1].split("RETRIEVED")[0].strip()[:200]
            print(f"\n[{r['task']}] Q: {q}\n--- GOLD ---\n{r['messages'][1]['content'][:600]}\n--- MODEL ---\n{p[:600]}")

    if args.merge_to:
        if not args.adapter:
            sys.exit("--merge-to needs --adapter")
        if args.quant != "none":
            print("[warn] merging into a 4-bit base is lossy; reload with --quant none for a clean merge")
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        Path(args.merge_to).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merge_to, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(args.merge_to)
        print(f"\nMerged model saved to {args.merge_to}")
        print("Use it by pointing src/llm.py (AVAILABLE_MODELS) at that folder, or serve it with vLLM.")


if __name__ == "__main__":
    main()
