import re

from kg_core.extractor_base import ENTITY_EXTRACTION_MAX_GLEANINGS, Extractor
from kg_core.prompts_general import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT
from kg_core.token_utils import num_tokens_from_string
from kg_core.utils import split_string_by_multi_markers

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


class GraphExtractor(Extractor):
    def __init__(
        self,
        llm_invoker,
        language: str | None = "English",
        entity_types: list[str] | None = None,
        max_gleanings: int | None = None,
    ):
        super().__init__(llm_invoker, language, entity_types)
        self._extraction_prompt = GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = max_gleanings if max_gleanings is not None else ENTITY_EXTRACTION_MAX_GLEANINGS
        self._prompt_variables = {
            "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
            "record_delimiter": DEFAULT_RECORD_DELIMITER,
            "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
            "entity_types": ",".join(self._entity_types),
        }

    async def _process_single_content(self, chunk_key_dp: tuple[str, str], chunk_seq: int, num_chunks: int, out_results):
        token_count = 0
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        variables = {
            **self._prompt_variables,
            "input_text": content,
        }
        hint_prompt = self._extraction_prompt.format(**variables)
        response = await self._chat(hint_prompt, [{"role": "user", "content": "Output:"}], {})
        token_count += num_tokens_from_string(hint_prompt + response)

        results = response or ""
        history = [{"role": "system", "content": hint_prompt}, {"role": "user", "content": response}]

        for i in range(self._max_gleanings):
            history.append({"role": "user", "content": CONTINUE_PROMPT})
            response = await self._chat("", history, {})
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + response)
            results += response or ""

            if i >= self._max_gleanings - 1:
                break
            history.append({"role": "assistant", "content": response})
            history.append({"role": "user", "content": LOOP_PROMPT})
            continuation = await self._chat("", history, {})
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + response)
            if continuation.strip().upper().startswith("N"):
                break
            history.append({"role": "assistant", "content": "Y"})

        records = split_string_by_multi_markers(
            results,
            [self._prompt_variables["record_delimiter"], self._prompt_variables["completion_delimiter"]],
        )
        rcds = []
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            rcds.append(record.group(1))
        records = rcds
        maybe_nodes, maybe_edges = self._entities_and_relations(chunk_key, records, self._prompt_variables["tuple_delimiter"])
        out_results.append((maybe_nodes, maybe_edges, token_count))
