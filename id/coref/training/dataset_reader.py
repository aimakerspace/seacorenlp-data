import collections
import json
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

from allennlp.data import DatasetReader, Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp_models.coref.util import make_coref_instance


ClusterId = int
SpanIndices = Tuple[int, int]


@DatasetReader.register('coref-id')
class IndonesianCorefDatasetReader(DatasetReader):
    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters

    def _read(self, file_path: str) -> Iterable[Instance]:
        documents = []

        # Read jsonl file
        with open(file_path, 'r') as f:
            for line in f:
                documents.append(json.loads(line))
            f.close()

        for sentences in documents:
            clusters: DefaultDict[ClusterId, List[SpanIndices]] = collections.defaultdict(list)

            tokens, corefs = sentences['tokens'], sentences["corefs"]
            tokens = [token.lower() for token in tokens]

            for coref in corefs:
                if 'label' not in coref:
                    continue
                cluster_id, start, end = coref['label'], coref['start'], coref['end']
                clusters[cluster_id].append((start, end))

            yield self.text_to_instance([tokens], list(clusters.values()))

    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[SpanIndices]]] = None,
    ) -> Instance:
        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            gold_clusters,
            self._wordpiece_modeling_tokenizer,
            self._max_sentences,
            self._remove_singleton_clusters,
        )
