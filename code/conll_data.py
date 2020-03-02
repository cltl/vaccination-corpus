from __future__ import annotations

import os
from graphviz import Digraph
from nltk.corpus import propbank as pb
from nltk.corpus import nombank as nb
import dataclasses as dc
from typing import Optional, List, Tuple, Dict

DICT_MODIFIERS = {
    "AM-LOC": "location",
    "AM-COM": "companion",
    "AM-TMP": "time",
    "AM-DIR": "direction",
    "AM-GOL": "goal",
    "AM-MNR": "manner",
    "AM-EXT": "extent",
    "AM-REC": "reflexive",
    "AM-PRD": "secondary predication",
    "AM-PNC": "purpose",
    "AM-CAU": "cause",
    "AM-DIS": "discourse marker",
    "AM-DSP": "direct speech",
    "AM-ADV": "adverb",
    "AM-ADJ": "adjective",
}

DICT_CORE_ROLES = {
    "A0": "agent, causer, experiencer",
    "A1": "patient, undergoer",
    "A2": "instrument, benefactive, attribute",
    "A3": "starting point, benefactive, attribute",
    "A4": "ending point",
}

POS_VERBS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
POS_NOUNS = ["NN", "NNS", "NNP", "NNPS"]
POS_ADV = ["RB", "RBR", "RBS"]
POS_ADJ = ["JJ", "JJR", "JJS"]

DEPREL_AUX = ["aux", "auxpass"]


@dc.dataclass
class Token:
    """Dataclass for a Token"""

    sent_id: str
    token_id: str
    word: str
    lemma: str
    pos: str
    head: str
    deprel: str
    offset_start: str
    offset_end: str
    apred: Optional[Dict[str, str]]
    event: Optional[str]
    claim: Optional[str]
    attr_content: Optional[str]
    attr_source: Optional[str]
    attr_cue: Optional[str]

    def is_imperative(self) -> bool:
        """Returns True if the token is a imperative verb in the sentence,
        False otherswise
        """
        if self.deprel == "ROOT" and self.token_id == "1" and self.pos in POS_VERBS:
            return True
        return False


@dc.dataclass
class Span:
    """Dataclass for a Span of Tokens"""

    tokens: List[Token]
    token_ids: List[str] = dc.field(init=False)
    sent_token_ids: List[Tuple[str, str]] = dc.field(init=False)
    words: List[str] = dc.field(init=False)
    lemmas: List[str] = dc.field(init=False)
    text: str = dc.field(init=False)

    def __post_init__(self) -> None:
        self.token_ids = [token.token_id for token in self.tokens]
        self.sent_token_ids = [(token.sent_id, token.token_id) for token in self.tokens]
        self.words = [token.word for token in self.tokens]
        self.lemmas = [token.lemma for token in self.tokens]
        self.text = " ".join(token.word for token in self.tokens)

    def get_token(self, token_id: str, sent_id: Optional[str] = None) -> Token:
        """Returns the Token in the Span with the specified token_id"""
        if sent_id is None:
            token = next(token for token in self.tokens if token.token_id == token_id)
        else:
            token = next(
                token
                for token in self.tokens
                if token.token_id == token_id and token.sent_id == sent_id
            )
        return token

    def add_token(self, token: Token) -> None:
        """Adds a single token to the Span"""
        self.tokens.append(token)
        self.tokens = sorted(
            self.tokens, key=lambda token: int(token.token_id)
        )  # sort by token id
        self.__post_init__()

    def add_tokens(self, tokens: List[Token]) -> None:
        """Adds a list of tokens to the Span"""
        self.tokens.extend(tokens)
        self.tokens = sorted(
            self.tokens, key=lambda token: int(token.token_id)
        )  # sort by token id
        self.__post_init__()

    def get_head_ids(self, initial_candidates: Optional[List[str]] = None):
        """Finds the list of ids of the heads of the given Span.
        Starts by looking at all token_ids, and recursively look
        at each head of token_ids to narrow down the list
        """
        candidates = []
        if initial_candidates is None:
            initial_candidates = [token.token_id for token in self.tokens]

        # if length of span is only one token, return list with single token_id
        if len(initial_candidates) == 1:
            return initial_candidates

        # else, check which tokens still have their head in the list
        for token in self.tokens:

            if (
                token.token_id in initial_candidates
                and token.head in initial_candidates
            ):
                candidates.append(token.head)

        # if no results found, we reached the end of the recursive loop
        # (cannot go any smaller)
        if not candidates:
            return initial_candidates

        # if candidates is not equal to initial_candidates, we didn't reach the end
        # of the recursive loop yet (can go even smaller)
        if not candidates == initial_candidates:
            candidates = self.get_head_ids(initial_candidates=candidates)

        return candidates

    def get_char_offset_in_context(self, context_span):
        """Returns the character offsets within a given context Span object"""

        # Get span (first word id, last word id)
        first_token_id = (self.tokens[0].sent_id, self.tokens[0].token_id)
        last_token_id = (self.tokens[-1].sent_id, self.tokens[-1].token_id)

        # Get character offset of tokens by counting characters
        char_count = 0
        for context_token in context_span.tokens:
            token_id = (context_token.sent_id, context_token.token_id)
            if token_id == first_token_id:
                offset_begin = char_count
            if token_id == last_token_id:
                offset_end = char_count + len(context_token.word)
            char_count += len(context_token.word) + 1  # count also space

        return offset_begin, offset_end

    def split_conjunction(self):
        all_phrases = []

        # check if first token is preposition: if so, include in all parts
        first_token = self.tokens[0]
        if first_token.pos == "IN" or first_token.deprel in [
            "prep",
            "xcomp",
            "case",
            "mark",
        ]:
            start_phrase = [first_token]
        else:
            start_phrase = []

        # split by conjunction
        phrase = start_phrase.copy()
        for token in self.tokens:
            if token.pos != "CC":
                phrase.append(token)
            else:
                if phrase:  # avoid empty spans (due to parsing errors)
                    all_phrases.append(phrase)
                phrase = start_phrase.copy()
        if phrase:  # avoid empty spans (due to parsing errors)
            all_phrases.append(phrase)
        return all_phrases

    def contains_lemma(self, lemma) -> bool:
        """Returns True if the Span contains the given lemma,
        False otherwise
        """
        for token in self.tokens:
            if token.lemma == lemma:
                return True
        return False


@dc.dataclass
class Document(Span):
    """Dataclass for a Document"""

    id: str
    sentences: List[Sentence] = dc.field(init=False)
    propositions: List[Proposition] = dc.field(init=False)
    attr_contents: List[Annotation] = dc.field(init=False)
    attr_sources: List[Annotation] = dc.field(init=False)
    attr_cues: List[Annotation] = dc.field(init=False)
    attr_relations: List[AttributionRelation] = dc.field(init=False)
    events: List[Annotation] = dc.field(init=False)
    claims: List[Annotation] = dc.field(init=False)

    def __post_init__(self):
        Span.__post_init__(self)
        self.sentences = self._get_sentences()
        self.propositions = self._get_propositions()
        self.attr_contents = self._get_annotations("attr_content")
        self.attr_sources = self._get_annotations("attr_source")
        self.attr_cues = self._get_annotations("attr_cue")
        self.attr_relations = self._get_attributions()
        self.events = self._get_annotations("event")
        self.claims = self._get_annotations("claim")

    def _get_sentences(self):
        """
        Iterates over all Tokens in Document and returns a list of Sentences
        """
        sentences: List[Sentence] = []
        sentence: List[Token] = []
        current_sent_id = self.tokens[0].sent_id
        for token in self.tokens:
            sent_id = token.sent_id
            if sent_id != current_sent_id:
                s = Sentence(tokens=sentence, sent_id=current_sent_id, document=self)
                sentences.append(s)
                # define new sentence
                current_sent_id = sent_id
                sentence = [token]
            else:
                sentence.append(token)
        # append last sentence
        if sentence not in sentences:
            s = Sentence(tokens=sentence, sent_id=current_sent_id, document=self)
            sentences.append(s)
        return sentences

    def _get_propositions(self):
        """
        Returns all Proposition objects found in the Sentences of Document
        """
        propositions = []
        for sent in self.sentences:
            propositions.extend(sent.propositions)
        return propositions

    def _get_annotations(self, name_column: str):
        """
        Returns the annotations in the specified column as Annotation objects
        """
        annotations = list()
        for index, B_token in enumerate(self.tokens):
            token_annotation = getattr(B_token, name_column)
            if token_annotation:

                # Get the B-elements
                B_elements = [
                    element
                    for element in token_annotation.split("#")
                    if element.startswith("B-")
                ]
                for B_element in B_elements:

                    # Get id and class of annotation
                    ann_id = B_element.split(":")[0].split("-")[-1]
                    ann_class = B_element.split("-")[1].lower()

                    # Create tuples (id, class) for relations and add to list
                    relations = []
                    if ":" in B_element:
                        for relation in B_element.split(":")[1].split("_"):
                            relation_id, relation_class = relation.split("-")
                            relations.append((relation_id, relation_class.lower()))

                    # Get the full span by finding I-tokens (and D- for discontinuous)
                    # with same id
                    ann_tokens = [B_token]
                    for I_token in self.tokens[index:]:
                        I_token_annotation = getattr(I_token, name_column)
                        if I_token_annotation:
                            I_elements = [
                                element.lstrip("I-").lstrip("D-")
                                for element in I_token_annotation.split("#")
                                if element.startswith(("I-", "D-"))
                            ]
                            if f"{ann_class}-{ann_id}" in I_elements:
                                ann_tokens.append(I_token)

                    # Create Annotation object and add to list
                    annotation = Annotation(
                        tokens=ann_tokens,
                        ann_class=ann_class,
                        ann_id=ann_id,
                        relations=relations,
                    )
                    annotations.append(annotation)

        return annotations

    def _get_attributions(self):
        """
        Returns the Attribution Relations in the Document as a list of
        AttributionRelation objects
        """

        attributions = []
        for content in self.attr_contents:
            cues = []
            sources = []
            for ann_id, ann_class in content.relations:
                if ann_class == "cue":
                    cue = next(
                        (cue for cue in self.attr_cues if cue.ann_id == ann_id), None,
                    )
                    if cue:
                        cues.append(cue)
                elif ann_class == "source":
                    source = next(
                        (
                            source
                            for source in self.attr_sources
                            if source.ann_id == ann_id
                        ),
                        None,
                    )
                    if source:
                        sources.append(source)
            attribution = AttributionRelation(content, cues=cues, sources=sources)
            attributions.append(attribution)

        return attributions

    def get_sentence(self, sent_id: str) -> Optional[Sentence]:
        """
        Returns the Sentence in the Document with the specified sent_id
        """
        sentence = next(
            (
                sentence
                for sentence in self.sentences
                if sentence.sent_id == str(sent_id)
            ),
            None,
        )
        return sentence

    def get_context(self, sent_id: str, context_window: int = 1) -> Tuple[Span, Span]:
        """
        Returns the previous and following sentences of a given Sentence
        within a specified context_window
        """
        last_sent_id = self.sentences[-1].sent_id  # get last sentence id
        first_sent_id = self.sentences[0].sent_id  # get first sentence id

        # Get previous sentences
        previous_context = []
        for window_id in reversed(range(1, context_window + 1)):
            previous_sent_id = str(int(sent_id) - window_id)
            if not int(previous_sent_id) < int(first_sent_id):
                previous_sent = next(
                    sent for sent in self.sentences if sent.sent_id == previous_sent_id
                )
                previous_context.append(previous_sent)
        tokens = [token for sent in previous_context for token in sent.tokens]
        previous_context_span = Span(tokens)

        # Get following sentences
        following_context = []
        for window_id in range(1, context_window + 1):
            following_sent_id = str(int(sent_id) + window_id)
            if not int(following_sent_id) > int(last_sent_id):
                following_sent = next(
                    sent for sent in self.sentences if sent.sent_id == following_sent_id
                )
                following_context.append(following_sent)
        tokens = [token for sent in following_context for token in sent.tokens]
        following_context_span = Span(tokens)

        return previous_context_span, following_context_span

    def get_relevant_sources(self, tokens, context_window=1):
        """
        Returns the sources of the Attribution Relation where 'tokens' is part
        of the content and the source is within the context_window
        """

        # Get valid sent ids (to retrieve only those sources
        # occuring in the context window)
        sent_id = tokens[0].sent_id
        valid_sent_ids = [sent_id]
        for x in range(1, context_window + 1):
            previous_sent_id = str(int(sent_id) + x)
            following_sent_id = str(int(sent_id) - x)
            valid_sent_ids.append(previous_sent_id)
            valid_sent_ids.append(following_sent_id)

        # Get all relevant sources; those where tokens is part of
        # attribution content + sources occurring in valid context window
        relevant_sources = []
        for attribution in self.attr_relations:
            if all([token in attribution.content.tokens for token in tokens]):
                for source in attribution.sources:
                    if (
                        source.tokens[0].sent_id in valid_sent_ids
                        and source.tokens[-1].sent_id in valid_sent_ids
                    ):  # assure source is entirely in context sentences
                        relevant_sources.append(source)

        return relevant_sources


@dc.dataclass
class Sentence(Span):

    sent_id: str
    document: Document
    id: str = dc.field(init=False)
    propositions: List[Proposition] = dc.field(init=False)

    def __post_init__(self):
        Span.__post_init__(self)
        self.id = self.document.id + "_" + self.sent_id
        self.propositions = self._get_propositions()

    def _get_semroles(self, apred_key: str) -> List[SemanticRole]:
        """
        Given the index of the predicate in the Sentence, returns the
        list of semantic roles of that predicate
        """
        semroles: List[SemanticRole] = []
        semrole_tokens: List[Token] = []
        for token in self.tokens:
            if not token.apred:
                continue

            tag = token.apred[apred_key]
            if tag == "B-V":
                continue

            elif tag.startswith("B"):
                if semrole_tokens:
                    semrole = SemanticRole(label=label, tokens=semrole_tokens)
                    semroles.append(semrole)
                label: str = tag.split("-", 1)[1].replace("ARG", "A")
                semrole_tokens = [token]

            elif tag == "O":
                if semrole_tokens:
                    semrole = SemanticRole(label=label, tokens=semrole_tokens)
                    semroles.append(semrole)
                semrole_tokens = []

            elif tag.startswith("I"):
                semrole_tokens.append(token)

        if semrole_tokens:
            semrole = SemanticRole(tokens=semrole_tokens, label=label)
            semroles.append(semrole)

        return semroles

    def _get_propositions(self):
        """
        Finds all semantic roles belonging to the predicates in Sentence and
        sets self.propositions to a list of Proposition objects
        """
        propositions = []
        apred_keys = self.tokens[0].apred.keys()
        for apred_key in apred_keys:

            # find predicate and determine if copular
            predicate = next(
                (token for token in self.tokens if token.apred[apred_key] == "B-V"),
                None,
            )
            if not predicate:
                continue

            # get semroles
            semroles = self._get_semroles(apred_key)

            # create proposition and add to list (only add those that
            # actually have semantic roles)
            if semroles:
                proposition = Proposition(
                    predicate=predicate, semroles=semroles, sentence=self,
                )
                propositions.append(proposition)

        return propositions

    def get_children(self, parent_token_id: str):
        children = []
        for token in self.tokens:
            if token.head == parent_token_id:
                children.append(token)
        return children

    def get_full_phrase(
        self, head_id: str, phrase: Optional[Span] = None, ids_to_ignore: List[str] = []
    ):
        """
        Recursively finds the dependents of the given head_id and returns
        the list of Tokens that is the complete phrase of head_id
        """

        # if starting the iterative process, initiate phrase_tokens
        if phrase is None:
            phrase = Span([token for token in self.tokens if token.token_id == head_id])

        # find dependents
        dependents = [
            token
            for token in self.tokens
            if all(
                [
                    token.head == head_id,
                    token not in phrase.tokens,
                    not (token.deprel == "aux" and token.head in ids_to_ignore),
                    token.token_id not in ids_to_ignore,
                ]
            )
        ]

        # add dependents to phrase_tokens and iteratively add dependents
        if dependents:
            phrase.add_tokens(dependents)
            for dep in dependents:
                dep_id = dep.token_id
                phrase = self.get_full_phrase(
                    dep_id, phrase=phrase, ids_to_ignore=ids_to_ignore
                )

        return phrase

    def get_auxiliaries(self, token_id: str, aux=None) -> List[Token]:
        """
        Returns the auxiliaries of a token specified by its token_id
        """
        if aux is None:
            aux = []

        # add auxiliaries of token_id
        for token in self.tokens:
            if all(
                [token.head == token_id, token.deprel in DEPREL_AUX, token not in aux]
            ):
                aux.append(token)

        # in case of conjunction, find aux of conjunction
        token = self.get_token(token_id)
        if token.deprel == "conj":
            conj = self.get_token(token.head)
            aux = self.get_auxiliaries(token_id=conj.token_id, aux=aux)
        return aux

    def is_passive(self, token_id: str) -> bool:
        """
        Returns True if the verb specified by token_id is in passive voice
        """
        passive = False
        for token in self.tokens:
            if token.head == token_id and token.deprel == "auxpass":
                passive = True

        # in case of conjunction, find aux of conjunction
        token = self.get_token(token_id)
        if token.deprel == "conj":
            conj = self.get_token(token.head)
            passive = self.is_passive(token_id=conj.token_id)

        return passive

    def check_conditionality_token(self, token_id: str) -> str:
        """
        Checks if the given token_id is part of a conditional construction
        in the sentence; returns 'condition', 'consequence' or 'NA'
        """
        for token in self.tokens:
            # head indicates 'if x' part, head_mate indicates 'then x' part
            if token.head == token_id and token.lemma == "if":
                return "condition"

            # following is not correct
            elif token.head == token_id and token.lemma == "if":
                return "consequence"
        return "NA"


@dc.dataclass
class SemanticRole(Span):
    label: str

    def __post_init__(self):
        Span.__post_init__(self)
        heads = [
            token for token in self.tokens if token.token_id in self.get_head_ids()
        ]
        self.heads = self._get_all_heads(heads)

    def _get_all_heads(self, heads: List[Token]):
        """Find conjunctions of heads and add to self.heads"""

        new_heads = heads.copy()
        for head in heads:
            conj_heads = [
                token
                for token in self.tokens
                if all([token.head == head.token_id, token.deprel == "conj"])
            ]
            new_heads.extend(conj_heads)

        return new_heads


@dc.dataclass
class Proposition:

    predicate: Token
    semroles: List[SemanticRole]
    sentence: Optional[Span] = None
    pred_roleset: Optional[str] = None
    id: str = dc.field(init=False)
    negated: bool = dc.field(init=False)
    modal: bool = dc.field(init=False)
    passive: bool = dc.field(init=False)
    copular: bool = dc.field(init=False)

    def __post_init__(self):
        self.id = self.sentence.id + "_" + self.predicate.token_id
        self.negated = self._contains_negation()
        self.modal = self._contains_modality()
        self.copular = self._is_copular()
        self.passive = self._is_passive()

    def _is_passive(self) -> bool:
        """Checks if the predicate is passive verb"""
        if self.sentence is not None:
            token_id = self.predicate.token_id
            for token in self.sentence.tokens:
                if token.head == token_id and token.deprel == "auxpass":
                    return True
        return False

    def _contains_negation(self) -> bool:
        """Checks if any of the semroles contains negation"""
        contains = False
        for semrole in self.semroles:
            first_token_deprel = semrole.tokens[0].deprel
            if semrole.label == "AM-NEG" or first_token_deprel == "neg":
                contains = True
        return contains

    def _contains_modality(self) -> bool:
        """Checks if any of the semroles is AM-MOD"""
        contains = any([semrole.label == "AM-MOD" for semrole in self.semroles])
        return contains

    def _is_copular(self) -> bool:
        if self.predicate.deprel == "cop":
            return True
        return False

    def generate_roleset(self) -> str:
        """Generates a roleset id for the predicate if it is not given
        (note: does not take WSD into account)"""
        if self.pred_roleset is not None:
            return self.pred_roleset
        roleset_id = f"{self.predicate.lemma}.01"
        pos = self.predicate.pos[0].lower()
        return f"{roleset_id}-{pos}"

    def get_nsubj_heads(self) -> List[Token]:
        """Returns the head of the semantic role which deprel is nsubj"""
        heads: List[Token] = []
        for semrole in self.semroles:
            for head in semrole.heads:
                if head.deprel in ["nsubj", "nmod", "nsubjpass"]:
                    if not semrole.label.startswith("R-"):
                        heads.append(head)
                    else:
                        label = semrole.label.lstrip("R-")
                        coref_heads = next(
                            (
                                semrole.heads
                                for semrole in self.semroles
                                if semrole.label == label
                            ),
                            None,
                        )
                        if coref_heads:
                            heads.extend(coref_heads)
        return heads

    def get_role_descriptions(self: Proposition) -> Dict[str, str]:
        """Retrieves the role descriptions of a specific roleset
        from PropBank/NomBank
        """
        if self.pred_roleset is None:
            pred_roleset = self.generate_roleset()
        else:
            pred_roleset = self.pred_roleset
        roleset_id, pos = pred_roleset.rsplit("-", 1)
        dict_semroles = DICT_MODIFIERS.copy()
        general_dict = DICT_CORE_ROLES.copy()
        if pos == "v":
            roleset = pb.roleset(roleset_id)
        elif pos == "n":
            roleset = nb.roleset(roleset_id)
        else:
            dict_semroles.update(general_dict)
            return dict_semroles
        for role in roleset.findall("roles/role"):
            number, description = role.attrib["n"], role.attrib["descr"]
            dict_semroles[f"A{number}"] = description
        dict_semroles.update(general_dict)
        return dict_semroles

    def create_graph(
        self,
        outdir: str = "graphs",
        graph_name: str = None,
        file_format: str = "pdf",
        exclude_labels: List[str] = [],
    ) -> None:
        """
        Creates a graph representation of a predicate with its semantic roles
        and writes it to a file
        """

        # Create graph with predicate node
        graph = Digraph(format=file_format)
        graph.node("pred", self.predicate.word)

        # Get the role descriptions for this predicate from PB/NB
        dict_semroles = self.get_role_descriptions()

        # Get predicate roleset (if necessary, create one)
        if self.pred_roleset is None:
            pred_roleset = self.generate_roleset()
        else:
            pred_roleset = self.pred_roleset

        # Iterate over all roles to get their label, text and general description
        for index, semrole in enumerate(self.semroles):
            role_label = semrole.label.lstrip("C-").lstrip("R-")
            if role_label in exclude_labels:
                continue
            elif role_label in dict_semroles:
                role_description = dict_semroles[role_label]
            else:
                role_description = ""

            role_text = insert_newlines(
                semrole.text, every=8
            )  # insert newline every 8 words (to reduce width nodes)

            # Create role node and edges
            role_label += index * " "  # trick to make duplicate edges possible
            graph.node(role_label, role_text)
            graph.edge("pred", role_label, label=role_description)

        # Write to file
        if not graph_name:
            graph_name = "#".join(
                [self.predicate.sent_id, self.predicate.token_id, pred_roleset]
            )
        os.makedirs(outdir, exist_ok=True)
        outfilepath = os.path.join(outdir, graph_name)
        graph.render(outfilepath, view=False, cleanup=True)


@dc.dataclass
class Annotation(Span):
    ann_class: str
    ann_id: str
    relations: List[Tuple[str, str]]

    def __post_init__(self) -> None:
        Span.__post_init__(self)


@dc.dataclass
class AttributionRelation:
    content: Annotation
    cues: List[Annotation]
    sources: List[Annotation]


def insert_newlines(string: str, every: int = 5) -> str:
    """Returns a string where \\n is inserted between every n words"""
    words = string.split()
    new_string = ""
    for i in range(0, len(words), every):
        new_string += " ".join(words[i : i + every]) + "\n"
    return new_string
