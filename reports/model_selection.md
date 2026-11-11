# Model selection notes

The default classifier is `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, running locally in float32. MiniLM remains the embedding model; DistilRoBERTa remains the emotion model.

During dashboard inspection, the first NLI MiniLM classifier assigned Non-fiction to several Agatha Christie mystery descriptions with low confidence. A smaller DeBERTa model improved some examples but still confused factual books. Expanded label descriptions also introduced an unwanted bias toward Non-fiction in that smaller model. The final DeBERTa base model with the simple Fiction/Non-fiction labels handled the inspected examples more sensibly.

The final spot-check included Spider's Web, Witness for the Prosecution & Selected Plays, Miss Marple, Partners in Crime, Murder in Mesopotamia, The Four Loves, The Problem of Pain, Empires of the Monsoon, The Gap Into Madness, Master of the Game, The Once and Future King, and Murder in LaMut. The model predicted Fiction for the story/play titles and Non-fiction for the three factual titles. The Four Loves was still uncertain (approximately 0.54), so the dashboard's uncertainty flag remains useful.

This was a small, deliberately selected development check. It is **not a held-out benchmark**, does not establish accuracy across the catalog, and should not be quoted as a performance percentage on a resume. A representative human-labelled sample is still needed.

The DeBERTa checkpoint declares float16 weights. The Transformers pipeline defaults to automatic dtype selection, which was slow on this CPU. Explicit float32 inference resolved that execution issue. Grouping similarly sized inputs reduces padding while restoring output order through cache keys. Both choices are visible in the source code.

The decision favors observed category quality while remaining fully local and free of API charges. There is no claim that this model is universally best. See the [model card](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) for the author's intended usage and training details.
