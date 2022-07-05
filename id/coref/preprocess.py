from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

import jsonlines
from tqdm import tqdm

MentionPair = Tuple[int, int]
MentionID = int
MentionLabel = str
CoreferenceLabel = str
Label = Union[MentionLabel, CoreferenceLabel]
MentionInformation = Dict[str, Any]
Cluster = Set[int]

MENTION_TYPES = {"PROPER", "PRONOUN", "NOUN", "VERB", "LIST"}
COREFERENCE_LINK_TYPES = {"IDENT", "APPOS", "ALIAS", "EXAPPOS"}


class CorefDataPreprocessor:
    def __init__(
        self,
        use_appos: bool,
        use_exappos: bool,
        use_aliases: bool,
        remove_singletons: bool
    ):
        # Flags
        self._use_appos: bool = use_appos
        self._use_exappos: bool = use_exappos
        self._use_aliases: bool = use_aliases
        self._remove_singletons: bool = remove_singletons

        # Paragraph-specific properties (Reset after every paragraph)
        self._text: str = ""
        self._tokens: List[str] = []
        self._labels: List[List[Label]] = []
        self._mention_dict: Dict[MentionID, MentionInformation] = {}
        self._clusters: List[Cluster] = []
        self._aliases: List[int] = []
        self._exappos_mentions: List[int] = []
        self._appos_mentions: Set[int] = set()

        # Overall statistics-related properties
        self._statistics: Dict[str, int] = {
          "paragraph_count": 0,
          "token_count": 0,
          "mention_count": 0,
          "cluster_count": 0,
          "singleton_count": 0
        }
        self._mention_types: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._link_types: Dict[str, int] = defaultdict(int)

    def _extract_mentions(self) -> None:
        tokens, labels = self._tokens, self._labels

        for i, label_list in enumerate(labels):
            if "_" in label_list:
                continue

            for label in label_list:
                # label_list example: ["IDENT[1_2]", "NOUN[2]"]
                if any(link_type in label for link_type in COREFERENCE_LINK_TYPES):
                    continue

                if len(label) == 0:
                    continue

                if label.split("[")[0] == "":
                    print(label)

                mention_id = int(label.split("[")[1][:-1])
                if mention_id not in self._mention_dict:
                    self._mention_dict[mention_id] = {"start": i}

                if (i == len(labels) - 1) or (label not in labels[i+1]):
                    self._mention_dict[mention_id]["end"] = i
                    start = self._mention_dict[mention_id]["start"]
                    text = tokens[start:i+1]
                    self._mention_dict[mention_id]["text"] = text

                    mention_type = label.split("[")[0]
                    self._mention_dict[mention_id]["type"] = mention_type

    def _handle_appositives(self, appositive_pairs: List[MentionPair]):
        """
        Appositives are labeled in the data in this manner:
        [Mas Ahmad]1 [umur 45 tahun]2 [Dia]3 --> APPOS[1_2] and IDENT[1_3]

        PSEUDOCODE
        ----------
        Track mention ID of appositive phrase (2nd element of mention pair)

        If APPOS is to be concatenated:
            Get end token index of appositive phrase and replace mention's end token index with it.
            Replace text of mention with the text of the full mention.
        Else:
            Do nothing.
        Finally:
            APPOS mentions and singletons will be dropped.
        """
        appos_clusters = self._group_mentions_into_clusters(appositive_pairs)
        clustered_appositive_pairs = [tuple(sorted(list(cluster))) for cluster in appos_clusters]

        for mention_pair in clustered_appositive_pairs:
            self._appos_mentions.update({mention_pair[1]})

            if not self._use_appos:
                continue

            else:
                span_start = self._mention_dict[mention_pair[0]]["start"]
                new_span_end = self._mention_dict[mention_pair[-1]]["end"]
                self._mention_dict[mention_pair[0]]["end"] = new_span_end
                self._mention_dict[mention_pair[0]]["text"] = self._tokens[span_start:new_span_end+1]

                # Deal with cases where () are not labelled under APPOS
                fused_text = self._mention_dict[mention_pair[0]]["text"]
                if fused_text.count("(") == fused_text.count(")") + 1:
                    if self._tokens[new_span_end+1] == ")":
                        self._mention_dict[mention_pair[0]]["end"] += 1
                        self._mention_dict[mention_pair[0]]["text"] = self._tokens[span_start:new_span_end+2]

    def _group_mentions_into_clusters(self, mention_pairs: List[MentionPair]) -> List[Cluster]:
        """
        [(1,2), (2,3), (4,5), (6,7), (5,9)] -> [{1,2,3}, {4,5,9}, {6,7}]
        """

        if len(mention_pairs) == 0:
            return []

        cluster_list = [set(mention_pairs[0])]

        for pair in mention_pairs[1:]:
            matched = False
            for cluster in cluster_list:
                if set(pair).intersection(cluster):
                    cluster.update(pair)
                    matched = True
                    break

            if not matched:
                cluster_list.append(set(pair))

        return cluster_list

    def _extract_coreference_links(self) -> None:
        """
        Extracts clusters from the data labels. Does not consider links without labels.
        """
        mention_pairs: List[MentionPair] = []
        appositive_pairs: List[MentionPair] = []

        labels = self._labels

        for label_list in labels:
            for label in label_list:
                if any(link_type in label for link_type in COREFERENCE_LINK_TYPES):
                    mention_pair = label.split("[")[1][:-1].split("_") # IDENT[1_2] => [1, 2]
                    label_class = label.split("[")[0]

                    if len(mention_pair) != 2:
                        print(f"Error: {label} does not contain a pair of mentions. Skipping...")
                        continue

                    self._link_types[label_class] += 1
                    mention_pair = (int(mention_pair[0]), int(mention_pair[1]))

                    if label_class == "APPOS":
                        appositive_pairs.append(mention_pair)

                    else:
                        mention_pairs.append(mention_pair)

                        # Keep track of mentions linked by ALIAS for removal
                        if not self._use_aliases:
                            if label_class == "ALIAS":
                                self._aliases.append(mention_pair[1])

                        # Keep track of mentions linked by EXAPPOS for removal
                        if not self._use_exappos:
                            if label_class == "EXAPPOS":
                                self._exappos_mentions.append(mention_pair[1])

        # Check if all mentions in coreference links are valid
        for mention_pair in mention_pairs:
            absent_mentions = [mention not in self._mention_dict.keys() for mention in mention_pair]
            if any(absent_mentions):
                print(f"{absent_mentions} in {mention_pair}")

        self._handle_appositives(appositive_pairs)

        # Regardless of whether APPOS is kept, drop the appositive mention
        # Unless the appositive mention happens to be coreferent with another mention
        for mention in self._appos_mentions:
            if mention not in {mention for mention_pair in mention_pairs for mention in mention_pair}:
                self._mention_dict.pop(mention)

        self._clusters = self._group_mentions_into_clusters(mention_pairs)

    def _assign_mentions_to_clusters(self) -> None:
        singletons: List[int] = []

        for mention_id, mention_info in self._mention_dict.items():
            matched = False
            for i, cluster in enumerate(self._clusters):
                if mention_id in cluster:
                    mention_info["label"] = i
                    matched = True
                    break

            if not matched:
                self._statistics["singleton_count"] += 1
                singletons.append(mention_id)

        if self._remove_singletons:
            for singleton in singletons:
                self._mention_dict.pop(singleton)

    def _remove_aliases(self) -> None:
        if len(self._aliases) > 0:
            for mention_id in self._aliases:
                self._mention_dict.pop(mention_id)
                self._clusters = [cluster - {mention_id} for cluster in self._clusters]

    def _remove_exappos(self) -> None:
        if len(self._exappos_mentions) > 0:
            for mention_id in self._exappos_mentions:
                self._mention_dict.pop(mention_id)
                self._clusters =  [cluster - {mention_id} for cluster in self._clusters]

    def get_paragraph_data(self, paragraph: str) -> None:
        tokens: List[str] = []
        labels: List[List[Label]] = []

        # 1. Split paragraph into lines
        lines = paragraph.strip().split("\n")

        # 2. Extract full paragraph text
        text = lines[0][6:] # Remove #Text=

        # 3. Extract tokens, mentions, and links
        for line in lines[1:]:
            columns = line.split("\t")
            if len(columns) == 1:
                continue

            # Tokens are in the 3rd column
            tokens.append(columns[2])

            # Labels are in the 4th column
            labels.append(columns[3].split("|"))

        self._text, self._tokens, self._labels = text, tokens, labels

    def get_coref_info(self) -> None:
        self._extract_mentions()
        self._extract_coreference_links()
        self._assign_mentions_to_clusters()

        if not self._use_aliases:
            self._remove_aliases()

        if not self._use_exappos:
            self._remove_exappos()

    def reset(self) -> None:
        self._text = ""
        self._tokens = []
        self._labels = []
        self._mention_dict = {}
        self._clusters = []
        self._aliases = []
        self._appos_mentions = set()

    def log_dataset_statistics(self) -> None:
        paragraph_count = self._statistics["paragraph_count"]
        token_count = self._statistics["token_count"]
        mention_count = self._statistics["mention_count"]
        cluster_count = self._statistics["cluster_count"]
        singleton_count = self._statistics["singleton_count"]

        print("\nStatistics:\n--------------------------")
        print(f"Number of paragraphs: {paragraph_count}")
        print(f"Number of tokens: {token_count}")
        print(f"Number of mentions: {mention_count}")
        print(f"Number of clusters: {cluster_count} \n")

        print(f"Average paragraph length: {(token_count / paragraph_count):.1f}")
        print(f"Average number of mentions per paragraph: {(mention_count / paragraph_count):.1f}")
        print(f"Average number of clusters per paragraph: {(cluster_count / paragraph_count):.1f}")
        print(f"Average cluster size: {(mention_count / cluster_count):.1f} \n")

        if self._remove_singletons:
            print(f"Singleton mentions removed: {singleton_count} \n")

        for link_type, count in self._link_types.items():
            print(f"{link_type}: {count} links")
        print("")

        for mention_type, statistics in self._mention_types.items():
            print(f"{mention_type}: {statistics['count']} mentions")

    def _accumulate_mention_type_counts(self):
        for mention in self._mention_dict.values():
            self._mention_types[mention["type"]]["count"] += 1

    def convert_tsv_to_jsonl(self, tsv_path: str, jsonl_path: str):
        """
        FROM:
        =================================================
        #Text=An example
        1-1 0-2 An _ _ _ _ _
        1-2 3-10 example NOUN[1] _ _ _ _

        TO:
        =================================================
        {"text": "...", "tokens": [...], "corefs": [...]}
        """
        output_data = []

        with open(tsv_path, "r") as tsv_file:
            data = tsv_file.read()

            paragraphs = data.split("\n\n")

            for paragraph in tqdm(paragraphs, desc="Parsing paragraphs:"):
                # 1. Extract features, labels and cluster information
                self.get_paragraph_data(paragraph)
                self.get_coref_info()

                # 2. Update statistics and overall data pool
                paragraph_dict = {
                  "text": self._text,
                  "tokens": self._tokens,
                  "corefs": list(self._mention_dict.values())
                }
                output_data.append(paragraph_dict)

                self._statistics["paragraph_count"] += 1
                self._statistics["token_count"] += len(self._tokens)
                self._statistics["mention_count"] += len(self._mention_dict.keys())
                self._statistics["cluster_count"] += len(self._clusters)

                self._accumulate_mention_type_counts()

                # 3. Reset paragraph-related properties for next paragraph
                self.reset()

            with jsonlines.open(jsonl_path, "w") as jsonl_file:
                print(f"Saving data to {jsonl_path} ...")
                jsonl_file.write_all(output_data)

            self.log_dataset_statistics()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_aliases", action="store_true")
    parser.add_argument("--use_appos", action="store_true")
    parser.add_argument("--use_exappos", action="store_true")
    parser.add_argument("--remove_singletons", action="store_true")
    args = parser.parse_args()

    preprocessor = CorefDataPreprocessor(
        use_appos=args.use_appos,
        use_exappos=args.use_exappos,
        use_aliases=args.use_aliases,
        remove_singletons=args.remove_singletons
    )
    preprocessor.convert_tsv_to_jsonl(args.input_path, args.output_path)
