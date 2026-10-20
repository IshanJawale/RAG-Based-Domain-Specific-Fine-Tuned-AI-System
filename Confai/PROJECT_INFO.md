# Project Title
**Domain-Adapted Intelligence on Commodity Hardware**  
*Rank-Stabilized LoRA with Federated Hybrid Retrieval*

---

# Project Info

General-purpose language models can be fluent yet unreliable on specialized, high-stakes tasks, while deploying larger models may require expensive hardware or sending sensitive information to cloud services. This project develops an end-to-end, locally deployable multimodal AI system that combines parameter-efficient adaptation, large-scale retrieval, and tool-using agents. Its initial application is Indian legal research, where responses need to be grounded in statutes and court judgments.

I fine-tuned Qwen3-VL-4B and Qwen3-VL-8B using Rank-Stabilized Low-Rank Adaptation (RS-LoRA) on 50,000 curated instruction–response pairs, training only around 0.7–0.8% of model parameters. With 4-bit NF4 quantization, the fine-tuned 8B model runs locally on an RTX 5060 laptop GPU with 8 GB VRAM, reaching a measured peak of 5.96 GB.

To ground responses in source material, the system combines LanceDB dense retrieval, BM25 lexical search, intent routing, and cross-encoder reranking over approximately 5.5 million judgment chunks and 99,000 statutory sections. A tool-using agent can invoke search utilities and deterministic calculations, then generate formatted, multi-page DOCX documents through a conversational interface.

In a rigorous benchmark evaluation on 100 strictly unseen held-out Indian legal questions (`val.jsonl`), domain fine-tuning produced dramatic accuracy gains: for the 4B model, perplexity dropped from 3.84 to 2.14 (-44.3%), ROUGE-2 surged from 9.33% to 14.85% (+59.2%), and BLEU-4 increased from 2.74 to 4.57 (+66.8%). For the 8B model, perplexity dropped from 3.63 to 2.06 (-43.3%), ROUGE-2 increased from 9.52% to 13.98% (+46.8%), and BLEU-4 increased from 2.90 to 4.11 (+41.7%), achieving 74.51% citation recall. The project demonstrates a practical architecture for sovereign, resource-efficient specialized AI by connecting lightweight fine-tuning with grounded federated retrieval and autonomous executable tools.
