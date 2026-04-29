### utils/nlp/few_shot_retriever.py
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.nlp.sentence_transformer import SentenceSimilarityModel


class FewShotRetriever:
    def __init__(self) -> None:
        self.model = SentenceSimilarityModel.get_instance()

    def semantic_search(
        self,
        query_vec: np.ndarray,
        references: List[str],
        k: int = 5,
    ) -> List[int]:
        """
        Compute cosine similarities between input query and reference list.

        :param query_vec: Query text string or embedding
        :param references: List of reference text strings
        :param k: Top-k results to return
        :return: List of indices corresponding to top-k most similar entries
        """
        ref_vecs = self.model.encode_sentences(references)
        query_vec = self.model.encode_sentence(query_vec)

        similarities = self.model.get_similar_ref(query_vec, ref_vecs)
        return np.argsort(similarities)[::-1][:k].tolist()

    def _load_reference_data(self, path: Path) -> Dict[str, str]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _format_few_shot(self, task_nl: str, task_data: dict, idx: int) -> str:
        return f"""
### **Example {idx}**

**Input**
```
{task_nl}
```

**Output**
```json
{json.dumps(task_data, indent=4)}
```
---
"""

    def generate_few_shot_prompts(
        self,
        input_nl: str,
        top_k: int,
        nl_task_db_path: Path = Path("assets/nl_task_db.json"),
        tasks_dir: Path = Path("assets/tasks"),
    ) -> str:
        """
        Generate few-shot prompt examples based on semantic similarity.

        :param input_nl: Input natural language query
        :param top_k: Number of similar examples to retrieve
        :param nl_task_db_path: Path to natural language → task file map
        :param tasks_dir: Directory where task JSON files are stored
        :return: Formatted string containing few-shot examples
        """
        refer_dict = self._load_reference_data(nl_task_db_path)
        refer_nls = list(refer_dict.keys())

        top_k_idx = self.semantic_search(input_nl, refer_nls, k=top_k)
        top_k_dict = {refer_nls[idx]: refer_dict[refer_nls[idx]] for idx in top_k_idx}

        prompt_blocks = []
        for i, (task_nl, file_name) in enumerate(top_k_dict.items(), start=1):
            with open(tasks_dir / file_name, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            prompt_blocks.append(self._format_few_shot(task_nl, task_data, i))

        return "\n".join(prompt_blocks)
