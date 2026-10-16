import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PoSSUM modular experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot",help="Run compact experts; reserve final test without scoring it")
    p.add_argument("input",help="Normalized JSONL records")
    p.add_argument("--output",default="runs/pilot")
    p.add_argument("--components",nargs="+",default=["bge","bert"],choices=["bge","bert","modernbert","gliner","gliner2"])
    p.add_argument("--device",choices=["cpu","mps"],default="cpu")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limit",type=int,default=800)
    p = sub.add_parser("score",help="Run one expert independently, caching raw scores/features")
    p.add_argument("component",choices=["bge","bert","modernbert","gliner","gliner2","mistral","qwen","mlx","openjev"])
    p.add_argument("input",help="JSONL with text field")
    p.add_argument("--output",required=True)
    p.add_argument("--model-id")
    p.add_argument("--device",default="cpu",choices=["cpu","mps"])
    args=parser.parse_args()
    if args.command == "pilot":
        from .pilot import run
        run(args.input,args.output,args.components,args.seed,args.limit,args.device)
        return
    from .data import read_jsonl
    rows=read_jsonl(args.input)
    kw={"model_id":args.model_id} if args.model_id else {}
    n=args.component
    if n == "bge":
        from .components.bge import BGE
        model=BGE(device=args.device,**kw)
    elif n in ("bert","modernbert"):
        from .components.bert import BERT
        model=BERT(device=args.device,**(kw or ({"model_id":"answerdotai/ModernBERT-base"} if n=="modernbert" else {})))
    elif n in ("gliner","gliner2"):
        from .components.gliner import GLiNER
        model=GLiNER(**(kw or ({"model_id":"fastino/gliner2-base-v1"} if n=="gliner2" else {})))
    elif n=="mistral":
        from .components.mistral import Mistral
        model=Mistral(device=args.device,**kw)
    elif n=="qwen":
        from .components.qwen import Qwen
        model=Qwen(device=args.device,**kw)
    elif n=="mlx":
        from .components.causal import MLXCausal
        if not args.model_id:
            parser.error("mlx requires --model-id with a pinned local MLX checkpoint")
        model=MLXCausal(args.model_id)
    else:
        from .components.openjev import OpenJev
        model=OpenJev()
    texts=[r["text"] for r in rows]
    import numpy as np
    result=model.encode(texts) if hasattr(model,"encode") else model.score(texts)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,values=result,ids=np.array([r.get("id",str(i)) for i,r in enumerate(rows)]))
    target.with_suffix(".json").write_text(json.dumps({"component":n,"model_id":args.model_id,"revision":getattr(model,"revision",None),"server_version":getattr(model,"version",None),"kind":"features" if hasattr(model,"encode") else "scores"},indent=2))

if __name__ == "__main__":
    main()
