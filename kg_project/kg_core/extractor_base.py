import asyncio
import logging
from collections import Counter, defaultdict

from kg_core.prompts_general import SUMMARIZE_DESCRIPTIONS_PROMPT
from kg_core.token_utils import truncate
from kg_core.utils import (
    chat_limiter,
    flat_uniq_list,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    split_string_by_multi_markers,
)

GRAPH_FIELD_SEP = "<SEP>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
ENTITY_EXTRACTION_MAX_GLEANINGS = 2


class Extractor:
    def __init__(self, llm_invoker, language: str | None = "English", entity_types: list[str] | None = None):
        self._llm = llm_invoker
        self._language = language
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.callback = None

    async def _chat(self, system, history, gen_conf=None):
        return await self._llm.async_chat(system, history, gen_conf or {})

    def _entities_and_relations(self, chunk_key: str, records: list, tuple_delimiter: str):
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        ent_types = [t.lower() for t in self._entity_types]
        for record in records:
            record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

            if_entities = handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None and if_entities.get("entity_type", "unknown").lower() in ent_types:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = handle_single_relationship_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        return dict(maybe_nodes), dict(maybe_edges)

    async def __call__(self, doc_id: str, chunks: list[str]):
        async def worker(chunk_key_dp: tuple[str, str], idx: int, total: int, out_results):
            await self._process_single_content(chunk_key_dp, idx, total, out_results)

        out_results = []
        tasks = [
            asyncio.create_task(worker((doc_id, ck), i, len(chunks), out_results))
            for i, ck in enumerate(chunks)
        ]
        await asyncio.gather(*tasks, return_exceptions=False)

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for m_nodes, m_edges, _token_count in out_results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        all_entities_data = []
        all_relationships_data = []

        for en_nm, ents in maybe_nodes.items():
            if not ents:
                continue
            entity_type = Counter([dp["entity_type"] for dp in ents]).most_common(1)[0][0]
            description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in ents])))
            description = await self._handle_entity_relation_summary(en_nm, description)
            all_entities_data.append(
                {
                    "entity_name": en_nm,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": flat_uniq_list(ents, "source_id"),
                }
            )

        for (src, tgt), rels in maybe_edges.items():
            weight = sum([edge.get("weight", 1.0) for edge in rels])
            description = GRAPH_FIELD_SEP.join(sorted(set([edge["description"] for edge in rels])))
            description = await self._handle_entity_relation_summary(f"{src} -> {tgt}", description)
            all_relationships_data.append(
                {
                    "src_id": src,
                    "tgt_id": tgt,
                    "description": description,
                    "keywords": flat_uniq_list(rels, "keywords"),
                    "weight": weight,
                    "source_id": flat_uniq_list(rels, "source_id"),
                }
            )

        if not all_entities_data and not all_relationships_data:
            logging.warning("No entities or relationships extracted.")

        return all_entities_data, all_relationships_data

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str) -> str:
        summary_max_tokens = 512
        use_description = truncate(description, summary_max_tokens)
        description_list = use_description.split(GRAPH_FIELD_SEP)
        if len(description_list) <= 12:
            return use_description
        prompt_template = SUMMARIZE_DESCRIPTIONS_PROMPT
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=description_list,
            language=self._language,
        )
        use_prompt = prompt_template.format(**context_base)
        async with chat_limiter:
            summary = await self._chat("", [{"role": "user", "content": use_prompt}], {})
        return summary
