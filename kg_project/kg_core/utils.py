import asyncio
import html
import re
import time
from collections import defaultdict

chat_limiter = asyncio.Semaphore(10)


def clean_str(value):
    if not isinstance(value, str):
        return value
    result = html.unescape(value.strip())
    return re.sub(r"[\"\x00-\x1f\x7f-\x9f]", "", result)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args)]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def handle_single_entity_extraction(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    return dict(
        entity_name=entity_name.upper(),
        entity_type=entity_type.upper(),
        description=entity_description,
        source_id=chunk_key,
    )


def handle_single_relationship_extraction(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_keywords = clean_str(record_attributes[4])
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    pair = sorted([source.upper(), target.upper()])
    return dict(
        src_id=pair[0],
        tgt_id=pair[1],
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=chunk_key,
        metadata={"created_at": time.time()},
    )


def flat_uniq_list(items, key):
    result = []
    for item in items:
        if isinstance(item, dict):
            values = item.get(key, [])
        else:
            values = getattr(item, key, [])
        if isinstance(values, list):
            result.extend(values)
        else:
            result.append(values)
    return sorted(set(result))


def get_from_to(node1, node2):
    return (node1, node2) if node1 < node2 else (node2, node1)


def merge_nodes_and_edges(maybe_nodes, maybe_edges):
    all_entities_data = []
    all_relationships_data = []
    for entity_name, entities in maybe_nodes.items():
        all_entities_data.append(
            {
                "entity_name": entity_name,
                "entity_type": entities[0]["entity_type"],
                "description": " ".join(sorted({e["description"] for e in entities})),
                "source_id": flat_uniq_list(entities, "source_id"),
            }
        )

    for (src, tgt), edges in maybe_edges.items():
        all_relationships_data.append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "weight": sum(edge.get("weight", 1.0) for edge in edges),
                "description": " ".join(sorted({e["description"] for e in edges})),
                "keywords": flat_uniq_list(edges, "keywords"),
                "source_id": flat_uniq_list(edges, "source_id"),
            }
        )

    return all_entities_data, all_relationships_data


def merge_entities_and_edges(maybe_nodes, maybe_edges):
    merged_nodes = defaultdict(list)
    merged_edges = defaultdict(list)
    for name, nodes in maybe_nodes.items():
        merged_nodes[name].extend(nodes)
    for key, edges in maybe_edges.items():
        merged_edges[tuple(sorted(key))].extend(edges)
    return merged_nodes, merged_edges
